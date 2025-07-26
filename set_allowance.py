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
        print("Polymarket client initialized.")
    except Exception as e:
        print(f"Error initializing client: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Set Allowance ---
    print("\nSetting unlimited USDC allowance for the Polymarket exchange.")
    print("This will require a transaction on the Polygon network and may take a moment...")

    try:
        # This function sends a transaction to the blockchain to approve spending
        response = client.set_usdc_allowance(MAX_UINT_256)
        tx_hash = response.get('transaction_hash')
        
        if not tx_hash:
            print(f"Error: Allowance transaction failed. Response: {response}", file=sys.stderr)
            sys.exit(1)
            
        print(f"Allowance transaction sent successfully! Hash: {tx_hash}")
        print("Waiting for transaction to be mined...")

        # Wait for the transaction to complete
        client.wait_for_transaction(tx_hash, timeout=300)
        
        print("\n✅ Success! Allowance has been set.")
        print("Your scrapers should now be able to place trades successfully.")

    except Exception as e:
        print(f"\n❌ An error occurred while setting the allowance: {e}", file=sys.stderr)
        print("Please ensure your wallet has enough MATIC for gas fees and try again.")
        sys.exit(1)

if __name__ == "__main__":
    set_unlimited_allowance() 