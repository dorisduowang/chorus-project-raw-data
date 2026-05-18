"""
Google Drive Sync - Upload daily scan snapshots to Google Drive.

Uploads snapshots to a 'chorus-sync' folder in Google Drive for access
by external servers.
"""

import json
import os
from pathlib import Path
from datetime import datetime

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Scopes needed for upload
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Paths
CHORUS_DIR = Path(__file__).parent.parent
TOKEN_FILE = CHORUS_DIR / 'gdrive_sync_token.json'

# Client secret: use env var, or auto-discover client_secret*.json in repo root
_cred_env = os.environ.get('GDRIVE_CLIENT_SECRET')
if _cred_env:
    CREDENTIALS_FILE = Path(_cred_env)
else:
    _matches = sorted(CHORUS_DIR.glob('client_secret*.json'))
    CREDENTIALS_FILE = _matches[0] if _matches else CHORUS_DIR / 'client_secret.json'
DATA_DIR = CHORUS_DIR / 'Data'

# Target folder name in Google Drive
GDRIVE_FOLDER_NAME = 'chorus-sync'


def get_credentials(interactive=True):
    """Get or refresh Google Drive credentials.

    Args:
        interactive: If True, prompt for auth code. If False, raise if no valid creds.
    """
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not interactive:
                raise RuntimeError(
                    f"No valid credentials. Run authenticate() first to set up:\n"
                    f"  python -c \"from ingest.gdrive_sync import authenticate; authenticate()\""
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            # For headless servers: print URL and prompt for code
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"\nVisit this URL to authorize:\n{auth_url}\n")
            code = input("Enter the authorization code: ")
            flow.fetch_token(code=code)
            creds = flow.credentials

        # Save credentials for next run
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())

    return creds


def authenticate():
    """Interactive authentication - run once to set up credentials."""
    print("Google Drive Authentication for Chorus Sync")
    print("=" * 50)
    get_credentials(interactive=True)
    print("\nAuthentication successful! Token saved to:", TOKEN_FILE)
    print("Daily sync will now work automatically.")


def get_or_create_folder(service, folder_name):
    """Get existing folder or create new one."""
    # Search for existing folder
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    folders = results.get('files', [])

    if folders:
        return folders[0]['id']

    # Create new folder
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder['id']


def upload_file(service, file_path, folder_id, filename=None):
    """Upload a file to the specified folder."""
    if filename is None:
        filename = Path(file_path).name

    # Check if file already exists in folder
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id)').execute()
    existing = results.get('files', [])

    file_metadata = {'name': filename}
    media = MediaFileUpload(str(file_path), mimetype='application/json')

    if existing:
        # Update existing file
        file_id = existing[0]['id']
        updated = service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        return updated['id'], 'updated'
    else:
        # Create new file
        file_metadata['parents'] = [folder_id]
        created = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        return created['id'], 'created'


def sync_snapshot(snapshot_path=None, verbose=True):
    """
    Sync a snapshot to Google Drive.

    Args:
        snapshot_path: Path to snapshot file. If None, syncs latest.
        verbose: Print progress messages.

    Returns:
        Dict with sync results.
    """
    if verbose:
        print("Syncing to Google Drive...")

    # Get credentials and build service
    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)

    # Get or create sync folder
    folder_id = get_or_create_folder(service, GDRIVE_FOLDER_NAME)
    if verbose:
        print(f"  Using folder: {GDRIVE_FOLDER_NAME}")

    results = {'uploaded': [], 'errors': []}

    # Determine which snapshot to upload
    if snapshot_path is None:
        # Find latest snapshot
        snapshots_dir = DATA_DIR / 'snapshots'
        snapshots = sorted(snapshots_dir.glob('snapshot_*.json'), reverse=True)
        if not snapshots:
            if verbose:
                print("  No snapshots found!")
            return results
        snapshot_path = snapshots[0]

    # Upload snapshot
    try:
        file_id, action = upload_file(service, snapshot_path, folder_id)
        results['uploaded'].append({
            'file': snapshot_path.name,
            'action': action,
            'drive_id': file_id
        })
        if verbose:
            print(f"  {action.capitalize()}: {snapshot_path.name}")
    except Exception as e:
        results['errors'].append({'file': snapshot_path.name, 'error': str(e)})
        if verbose:
            print(f"  Error uploading {snapshot_path.name}: {e}")

    # Also upload current state file
    current_state = DATA_DIR / 'jevans_resources.json'
    if current_state.exists():
        try:
            file_id, action = upload_file(service, current_state, folder_id)
            results['uploaded'].append({
                'file': current_state.name,
                'action': action,
                'drive_id': file_id
            })
            if verbose:
                print(f"  {action.capitalize()}: {current_state.name}")
        except Exception as e:
            results['errors'].append({'file': current_state.name, 'error': str(e)})
            if verbose:
                print(f"  Error uploading {current_state.name}: {e}")

    # Upload change log
    change_log = DATA_DIR / 'change_log.json'
    if change_log.exists():
        try:
            file_id, action = upload_file(service, change_log, folder_id)
            results['uploaded'].append({
                'file': change_log.name,
                'action': action,
                'drive_id': file_id
            })
            if verbose:
                print(f"  {action.capitalize()}: {change_log.name}")
        except Exception as e:
            results['errors'].append({'file': change_log.name, 'error': str(e)})
            if verbose:
                print(f"  Error uploading {change_log.name}: {e}")

    if verbose:
        print(f"  Sync complete: {len(results['uploaded'])} files")

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Sync snapshots to Google Drive')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    parser.add_argument('--snapshot', '-s', type=str, help='Specific snapshot to sync')
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot) if args.snapshot else None
    sync_snapshot(snapshot_path=snapshot_path, verbose=not args.quiet)


if __name__ == '__main__':
    main()
