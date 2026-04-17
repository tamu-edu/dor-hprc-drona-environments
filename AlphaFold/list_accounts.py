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
    try:
        # Try executing external command and get the output (list of accounts)
        output = subprocess.check_output(["/sw/local/bin/myproject", "list_accounts"], stderr=subprocess.STDOUT)
        print("check 1")
        print(output)
        output = output.decode('utf-8')  # Decode byte string to Unicode string
        output = output.split('\n')  # Split output into lines
        output = output[5:]  # Skip the first 5 lines (assumed to be headers)
        print("check 2")
        accounts = []
        # Iterate through each line of output
        for line in output:
            if '|' in line:
                print(line)
                line = line.strip('|')  # Remove leading/trailing '|'
                data = line.split('|')  # Split line into fields by '|'
                data = [x.strip() for x in data]  # Remove extra spaces around fields
                account = account_from_list(data)  # Create Account object from the line data
                accounts.append(account)  # Add the account to the list

        # Convert list of Account objects to JSON format and print it
        json_data = json.dumps([ob.__dict__ for ob in accounts], indent=2)
        print(json_data)

    except subprocess.CalledProcessError as e:
        # Handle the error when the external command fails
        print(f"THIS Command failed with exit status {e.returncode}.\n Error output: {e.output.decode()}")

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
