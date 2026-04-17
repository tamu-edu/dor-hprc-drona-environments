#!/usr/bin/env python3
import subprocess
import json
import argparse

# Class to represent an account with 'value' as account number and 'label' as account number with balance
class Account:
    def __init__(self, account_no, balance):
        self.value = account_no
        self.label = f"{account_no}  ( {balance} )"

# Function to create an Account object from a list of data
def account_from_list(data):
    account_no = data[0]
    balance = data[5]
    return Account(account_no, balance)  # Return an Account object

# Function to list accounts and print them in JSON format
def list_accounts():
        # Try executing external command and get the output (list of accounts)
    output = subprocess.check_output(["/sw/local/bin/myproject", "list_accounts"], stderr=subprocess.STDOUT)
    print("check 1")
    print(output)


# Main function to handle argument parsing and calling appropriate functions
def main():
    # Create an argument parser object
    parser = argparse.ArgumentParser(description="Account listing utility")

    # Parse the arguments passed to the script
    args = parser.parse_args()

    # Call list_accounts function directly
    list_accounts()

# Entry point of the script
if __name__ == "__main__":
    main()
