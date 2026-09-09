# Security and Local Configuration

## Environment variables
See `config/.env.example`.

Never commit:
- Copernicus credentials
- OAuth tokens
- local model secrets
- access keys

## Local-only default
Bind development server to localhost by default.

## File security
Validate file paths before streaming artifacts.
Never accept arbitrary filesystem paths from the browser.

## Resource limits
Set configurable limits:
- maximum AOI size
- maximum raster pixels per request
- maximum concurrent jobs = 1 for MVP
- maximum upload size

## Reproducibility
Every job manifest must record:
- git commit hash if available
- Python version
- torch version
- SEN2SR package version
- Trust Head model version
- input scene id
- output artifact checksums
