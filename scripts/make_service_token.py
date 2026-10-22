"""
Mint the service token the Supabase Edge Function uses to call this API.

The token is long-lived on purpose: it is a server-to-server credential held in
Supabase's secret store, not a user session. Re-run this after rotating
JWT_SECRET, and set the new value with:

    supabase secrets set OSHO_API_TOKEN=<token>

Usage:
    JWT_SECRET=<your secret> python scripts/make_service_token.py [--years 10]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

from jose import jwt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=10, help="validity in years")
    parser.add_argument("--subject", default="supabase-edge-function")
    args = parser.parse_args()

    secret = os.environ.get("JWT_SECRET", "")
    if not secret or secret == "change_me_in_production":
        print(
            "JWT_SECRET is not set. Use the same value you set on the API host:\n"
            "  JWT_SECRET=<secret> python scripts/make_service_token.py",
            file=sys.stderr,
        )
        return 1

    expires = datetime.now(timezone.utc) + timedelta(days=365 * args.years)
    token = jwt.encode(
        {"sub": args.subject, "exp": expires},
        secret,
        algorithm=os.environ.get("JWT_ALGORITHM", "HS256"),
    )
    print(token)
    print(f"\n# subject={args.subject} expires={expires:%Y-%m-%d}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
