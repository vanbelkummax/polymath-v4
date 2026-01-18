#!/usr/bin/env python3
"""
Code Sandbox - Repository Verification

Verifies if repositories actually run by:
1. Cloning to temp directory
2. Creating virtual environment
3. Installing dependencies
4. Running tests (if available)

Safety: Uses virtual environments with timeouts. Docker support optional.

Usage:
    python scripts/verify_repos.py --limit 10 --language Python
    python scripts/verify_repos.py --verify-repo owner/name
    python scripts/verify_repos.py --stats
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Timeouts (seconds)
CLONE_TIMEOUT = 60
INSTALL_TIMEOUT = 300
TEST_TIMEOUT = 180

# Max repo size to clone (MB)
MAX_REPO_SIZE_MB = 500


# ============================================================================
# Verification Functions
# ============================================================================

def run_command(
    cmd: List[str],
    cwd: Path = None,
    timeout: int = 60,
    env: dict = None
) -> Tuple[bool, str, str]:
    """
    Run a command with timeout.

    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or os.environ.copy()
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def clone_repo(repo_url: str, dest: Path, branch: str = None) -> Tuple[bool, str]:
    """Clone repository to destination."""
    cmd = ['git', 'clone', '--depth', '1']
    if branch:
        cmd.extend(['-b', branch])
    cmd.extend([repo_url, str(dest)])

    success, stdout, stderr = run_command(cmd, timeout=CLONE_TIMEOUT)
    return success, stderr if not success else stdout


def detect_python_project(repo_path: Path) -> Dict:
    """
    Detect Python project type and dependencies.

    Returns:
        Dict with package_manager, install_cmd, test_cmd
    """
    info = {
        'is_python': False,
        'package_manager': None,
        'install_cmd': None,
        'test_cmd': None,
        'python_version': None
    }

    # Check for Python indicators
    py_files = list(repo_path.glob('**/*.py'))[:10]
    if not py_files:
        return info

    info['is_python'] = True

    # Detect package manager
    if (repo_path / 'pyproject.toml').exists():
        info['package_manager'] = 'pip'
        info['install_cmd'] = ['pip', 'install', '-e', '.']

        # Check for poetry
        with open(repo_path / 'pyproject.toml') as f:
            content = f.read()
            if '[tool.poetry]' in content:
                info['package_manager'] = 'poetry'
                info['install_cmd'] = ['poetry', 'install']
            elif '[project]' in content:
                info['install_cmd'] = ['pip', 'install', '-e', '.']

    elif (repo_path / 'setup.py').exists():
        info['package_manager'] = 'pip'
        info['install_cmd'] = ['pip', 'install', '-e', '.']

    elif (repo_path / 'requirements.txt').exists():
        info['package_manager'] = 'pip'
        info['install_cmd'] = ['pip', 'install', '-r', 'requirements.txt']

    elif (repo_path / 'environment.yml').exists():
        info['package_manager'] = 'conda'
        info['install_cmd'] = ['conda', 'env', 'create', '-f', 'environment.yml']

    # Detect test framework
    if (repo_path / 'pytest.ini').exists() or (repo_path / 'pyproject.toml').exists():
        info['test_cmd'] = ['pytest', '-x', '--tb=short', '-q']
    elif (repo_path / 'tests').is_dir():
        info['test_cmd'] = ['python', '-m', 'pytest', '-x', '--tb=short', '-q']
    elif (repo_path / 'test').is_dir():
        info['test_cmd'] = ['python', '-m', 'pytest', 'test/', '-x', '--tb=short', '-q']

    return info


def create_venv(venv_path: Path) -> Tuple[bool, str]:
    """Create a virtual environment."""
    success, stdout, stderr = run_command(
        [sys.executable, '-m', 'venv', str(venv_path)],
        timeout=60
    )
    return success, stderr if not success else "Virtual environment created"


def get_venv_python(venv_path: Path) -> str:
    """Get path to Python in virtual environment."""
    if os.name == 'nt':
        return str(venv_path / 'Scripts' / 'python.exe')
    return str(venv_path / 'bin' / 'python')


def verify_single_repo(
    repo_url: str,
    owner: str,
    name: str,
    branch: str = 'main'
) -> Dict:
    """
    Verify a single repository.

    Returns:
        Dict with verification results
    """
    result = {
        'repo_url': repo_url,
        'owner': owner,
        'name': name,
        'verification_status': 'unverified',
        'install_success': None,
        'tests_success': None,
        'log': [],
        'verified_at': datetime.now().isoformat()
    }

    # Create temp directory
    with tempfile.TemporaryDirectory(prefix='polymath_verify_') as tmpdir:
        tmpdir = Path(tmpdir)
        repo_path = tmpdir / name
        venv_path = tmpdir / 'venv'

        # Clone
        result['log'].append(f"Cloning {repo_url}...")
        success, msg = clone_repo(repo_url, repo_path, branch)
        if not success:
            result['verification_status'] = 'clone_failed'
            result['log'].append(f"Clone failed: {msg[:500]}")
            return result
        result['log'].append("Clone successful")

        # Detect project type
        project_info = detect_python_project(repo_path)

        if not project_info['is_python']:
            result['verification_status'] = 'not_python'
            result['log'].append("Not a Python project")
            return result

        if not project_info['install_cmd']:
            result['verification_status'] = 'no_install_method'
            result['log'].append("No installation method detected")
            return result

        result['log'].append(f"Detected: {project_info['package_manager']}")

        # Create virtual environment
        result['log'].append("Creating virtual environment...")
        success, msg = create_venv(venv_path)
        if not success:
            result['verification_status'] = 'venv_failed'
            result['log'].append(f"Venv creation failed: {msg}")
            return result

        venv_python = get_venv_python(venv_path)
        venv_pip = str(venv_path / 'bin' / 'pip') if os.name != 'nt' else str(venv_path / 'Scripts' / 'pip.exe')

        # Upgrade pip
        run_command([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'], timeout=60)

        # Install dependencies
        result['log'].append(f"Installing: {' '.join(project_info['install_cmd'])}")

        # Modify install command to use venv pip
        install_cmd = project_info['install_cmd'].copy()
        if install_cmd[0] == 'pip':
            install_cmd[0] = venv_pip

        success, stdout, stderr = run_command(
            install_cmd,
            cwd=repo_path,
            timeout=INSTALL_TIMEOUT
        )

        result['install_success'] = success
        if not success:
            result['verification_status'] = 'install_failed'
            result['log'].append(f"Install failed: {stderr[:500]}")
            return result
        result['log'].append("Installation successful")

        # Run tests if available
        if project_info['test_cmd']:
            result['log'].append(f"Running tests: {' '.join(project_info['test_cmd'])}")

            # Install pytest in venv
            run_command([venv_pip, 'install', 'pytest'], timeout=60)

            test_cmd = project_info['test_cmd'].copy()
            if test_cmd[0] == 'python':
                test_cmd[0] = venv_python
            elif test_cmd[0] == 'pytest':
                test_cmd[0] = str(venv_path / 'bin' / 'pytest') if os.name != 'nt' else str(venv_path / 'Scripts' / 'pytest.exe')

            success, stdout, stderr = run_command(
                test_cmd,
                cwd=repo_path,
                timeout=TEST_TIMEOUT
            )

            result['tests_success'] = success
            if success:
                result['log'].append("Tests passed")
                result['verification_status'] = 'verified'
            else:
                result['log'].append(f"Tests failed: {stderr[:300]}")
                result['verification_status'] = 'tests_failed'
        else:
            result['log'].append("No tests detected")
            result['verification_status'] = 'no_tests'
            result['tests_success'] = None

    return result


def update_repo_verification(conn, repo_id: str, result: Dict):
    """Update repository verification status in database."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE repositories SET
            verification_status = %s,
            verification_log = %s,
            last_verified_at = NOW(),
            install_success = %s,
            tests_success = %s
        WHERE repo_id = %s
    """, (
        result['verification_status'],
        '\n'.join(result['log']),
        result['install_success'],
        result['tests_success'],
        repo_id
    ))
    conn.commit()


# ============================================================================
# Main Logic
# ============================================================================

def verify_repos(
    conn,
    limit: int = 10,
    language: str = 'Python',
    min_stars: int = 0
) -> Dict:
    """
    Verify multiple repositories.

    Returns:
        Summary dict
    """
    cur = conn.cursor()

    # Get unverified repos
    cur.execute("""
        SELECT repo_id, repo_url, owner, name, default_branch, stars
        FROM repositories
        WHERE verification_status = 'unverified'
        AND language ILIKE %s
        AND (stars IS NULL OR stars >= %s)
        ORDER BY stars DESC NULLS LAST
        LIMIT %s
    """, (f'%{language}%', min_stars, limit))

    repos = cur.fetchall()
    logger.info(f"Found {len(repos)} unverified {language} repos to verify")

    results = {
        'total': len(repos),
        'verified': 0,
        'install_failed': 0,
        'tests_failed': 0,
        'no_tests': 0,
        'other': 0,
        'repos': []
    }

    for repo_id, repo_url, owner, name, branch, stars in repos:
        logger.info(f"Verifying {owner}/{name} ({stars or 0} stars)...")

        result = verify_single_repo(repo_url, owner, name, branch or 'main')
        update_repo_verification(conn, str(repo_id), result)

        results['repos'].append({
            'name': f"{owner}/{name}",
            'status': result['verification_status'],
            'stars': stars
        })

        # Update counters
        status = result['verification_status']
        if status == 'verified':
            results['verified'] += 1
        elif status == 'install_failed':
            results['install_failed'] += 1
        elif status == 'tests_failed':
            results['tests_failed'] += 1
        elif status == 'no_tests':
            results['no_tests'] += 1
        else:
            results['other'] += 1

        logger.info(f"  → {result['verification_status']}")

    return results


def get_verification_stats(conn) -> Dict:
    """Get verification statistics."""
    cur = conn.cursor()

    cur.execute("""
        SELECT verification_status, COUNT(*) as cnt
        FROM repositories
        GROUP BY verification_status
        ORDER BY cnt DESC
    """)

    status_dist = dict(cur.fetchall())

    cur.execute("""
        SELECT COUNT(*) FROM repositories WHERE install_success = TRUE
    """)
    install_success = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM repositories WHERE tests_success = TRUE
    """)
    tests_success = cur.fetchone()[0]

    cur.execute("""
        SELECT owner || '/' || name as repo, stars, verification_status
        FROM repositories
        WHERE verification_status = 'verified'
        ORDER BY stars DESC NULLS LAST
        LIMIT 10
    """)
    top_verified = [
        {'repo': row[0], 'stars': row[1], 'status': row[2]}
        for row in cur.fetchall()
    ]

    return {
        'status_distribution': status_dist,
        'install_success_count': install_success,
        'tests_success_count': tests_success,
        'top_verified': top_verified
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Verify repository quality')
    parser.add_argument('--limit', type=int, default=10, help='Max repos to verify')
    parser.add_argument('--language', default='Python', help='Filter by language')
    parser.add_argument('--min-stars', type=int, default=0, help='Minimum stars')
    parser.add_argument('--verify-repo', help='Verify specific repo (owner/name)')
    parser.add_argument('--stats', action='store_true', help='Show verification statistics')
    parser.add_argument('--output', '-o', type=Path, help='Output JSON file')

    args = parser.parse_args()

    conn = psycopg2.connect(config.POSTGRES_DSN)

    if args.verify_repo:
        # Verify single repo
        owner, name = args.verify_repo.split('/')
        repo_url = f"https://github.com/{owner}/{name}"

        logger.info(f"Verifying {repo_url}...")
        result = verify_single_repo(repo_url, owner, name)

        print(f"\n{'='*60}")
        print(f"VERIFICATION: {owner}/{name}")
        print(f"{'='*60}")
        print(f"Status: {result['verification_status']}")
        print(f"Install: {'✓' if result['install_success'] else '✗'}")
        print(f"Tests: {'✓' if result['tests_success'] else ('✗' if result['tests_success'] is False else 'N/A')}")
        print("\nLog:")
        for line in result['log']:
            print(f"  {line}")

    elif args.stats:
        stats = get_verification_stats(conn)

        print(f"\n{'='*60}")
        print("VERIFICATION STATISTICS")
        print(f"{'='*60}")
        print("\nStatus Distribution:")
        for status, count in stats['status_distribution'].items():
            print(f"  {status}: {count}")

        print(f"\nInstall Success: {stats['install_success_count']}")
        print(f"Tests Success: {stats['tests_success_count']}")

        if stats['top_verified']:
            print("\nTop Verified Repos:")
            for r in stats['top_verified']:
                print(f"  [{r['stars'] or 0} ★] {r['repo']}")

    else:
        results = verify_repos(
            conn,
            limit=args.limit,
            language=args.language,
            min_stars=args.min_stars
        )

        print(f"\n{'='*60}")
        print("VERIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Verified: {results['verified']}")
        print(f"Install failed: {results['install_failed']}")
        print(f"Tests failed: {results['tests_failed']}")
        print(f"No tests: {results['no_tests']}")
        print(f"Other: {results['other']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

    conn.close()


if __name__ == '__main__':
    main()
