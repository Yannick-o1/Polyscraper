import os
import sys
from py_clob_client.client import ClobClient

def debug_client_methods():
    """
    Initializes the ClobClient and prints all its available methods
    to help identify the correct function for setting allowances.
    """
    # --- Load Credentials ---
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")

    if not private_key or not proxy_address:
        print("Error: POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS environment variables must be set.", file=sys.stderr)
        sys.exit(1)

    print("Credentials loaded. Initializing client...")

    # --- Initialize Client ---
    try:
        client = ClobClient(
            "https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=1,
            funder=proxy_address
        )
        print("Client initialized successfully.")
    except Exception as e:
        print(f"Error initializing client: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Print all methods and attributes ---
    print("\n--- Available methods and attributes on the ClobClient object ---")
    # We filter out the 'private' methods for clarity
    available_methods = [method for method in dir(client) if not method.startswith('_')]
    for method in available_methods:
        print(method)
    print("\n--- End of list ---")
    print("\nPlease share this list. It will help find the correct function to set the token allowance.")

if __name__ == "__main__":
    debug_client_methods() 