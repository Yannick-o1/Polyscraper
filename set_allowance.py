import os
import sys
from py_clob_client.client import ClobClient

# The constant MAX_UINT_256 might not be in all versions of the library.
# We define it manually here. It's the standard value for an "unlimited" allowance.
MAX_UINT_256 = 2**256 - 1


def set_unlimited_allowance():
    """
    Initializes the ClobClient and sets an unlimited USDC allowance
    for the Polymarket exchange contract. This is a one-time operation.
    """
    # --- Load Credentials ---
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")

    if not private_key or not proxy_address:
        print("Error: POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS environment variables must be set.", file=sys.stderr)
        sys.exit(1)

    print("Credentials loaded successfully.")
    print(f"Funder Address: {proxy_address}")

    # --- Initialize Client ---
    try:
        client = ClobClient(
            "https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=1,
            funder=proxy_address
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        print("Polymarket client initialized and API credentials set.")
    except Exception as e:
        print(f"Error initializing client: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Set Allowance ---
    print("\nSetting unlimited USDC allowance for the Polymarket exchange.")
    print("This will require a transaction on the Polygon network if not already set.")
    print("Please wait...")

    try:
        # This function checks the current allowance and sends a transaction
        # to the blockchain to approve spending if the allowance is insufficient.
        response = client.update_balance_allowance()
        
        # This function may not return a transaction hash if the allowance is already sufficient.
        # We check the response to see what happened.
        if "transaction_hash" in response:
            tx_hash = response.get('transaction_hash')
            print(f"Allowance transaction sent successfully! Hash: {tx_hash}")
            print("Waiting for transaction to be mined...")
            client.wait_for_transaction(tx_hash, timeout=300)
            print("\n✅ Success! New allowance has been set.")
        elif response.get("allowance") == str(MAX_UINT_256) and response.get("balance") is not None:
             print("\n✅ Success! Your USDC allowance is already set to the maximum.")
        else:
            print(f"An unexpected response was received: {response}", file=sys.stderr)
            sys.exit(1)

        print("Your scrapers should now be able to place trades successfully.")

    except Exception as e:
        print(f"\n❌ An error occurred while setting the allowance: {e}", file=sys.stderr)
        print("Please ensure your wallet has enough MATIC for gas fees and try again.")
        sys.exit(1)

if __name__ == "__main__":
    set_unlimited_allowance() 