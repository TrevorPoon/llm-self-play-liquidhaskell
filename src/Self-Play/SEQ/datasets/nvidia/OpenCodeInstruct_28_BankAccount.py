from typing import *

class BankAccount:
    def __init__(self, account_holder_name, account_number):
        """
        Initialize a new bank account with the given account holder's name and account number.
        
        :param account_holder_name: The name of the account holder.
        :param account_number: The account number.
        """
        self.account_holder_name = account_holder_name
        self.account_number = account_number
        self.balance = 0
        self.transactions = []

    def deposit(self, amount):
        """
        Deposit a specified amount into the account.
        
        :param amount: The amount to deposit.
        :raises ValueError: If the amount is not positive.
        """
        if amount > 0:
            self.balance += amount
            self.transactions.append(f"Deposited: {amount}")
        else:
            raise ValueError("Deposit amount must be positive.")

    def withdraw(self, amount):
        """
        Withdraw a specified amount from the account.
        
        :param amount: The amount to withdraw.
        :raises ValueError: If the amount is not positive.
        :raises ValueError: If the amount exceeds the current balance.
        """
        if amount > self.balance:
            print("Insufficient funds. Withdrawal not allowed.")
        elif amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        else:
            self.balance -= amount
            self.transactions.append(f"Withdrew: {amount}")

    def get_balance(self):
        """
        Get the current balance of the account.
        
        :return: The current balance.
        """
        return self.balance

    def get_transactions(self):
        """
        Get a list of all transactions.
        
        :return: A list of transaction strings.
        """
        return self.transactions

    def get_statement(self):
        """
        Get a formatted statement of the account.
        
        :return: A formatted string with account details and transactions.
        """
        statement = (f"Account Holder: {self.account_holder_name}\n"
                     f"Account Number: {self.account_number}\n"
                     f"Current Balance: {self.balance}\n"
                     "Transactions:\n")
        for transaction in self.transactions:
            statement += f"  {transaction}\n"
        return statement

### Unit tests below ###
def check(candidate):
    assert BankAccount("John Doe", "123456789").get_balance() == 0
    assert BankAccount("John Doe", "123456789").get_transactions() == []
    account = BankAccount("John Doe", "123456789")
    account.deposit(100)
    assert account.get_balance() == 100
    account = BankAccount("John Doe", "123456789")
    account.deposit(100)
    account.deposit(50)
    assert account.get_balance() == 150
    account = BankAccount("John Doe", "123456789")
    account.deposit(100)
    account.withdraw(50)
    assert account.get_balance() == 50
    account = BankAccount("John Doe", "123456789")
    account.deposit(100)
    account.withdraw(150)
    assert account.get_balance() == 100
    account = BankAccount("John Doe", "123456789")
    account.deposit(100)
    account.withdraw(50)
    assert account.get_transactions() == ["Deposited: 100", "Withdrew: 50"]
    account = BankAccount("John Doe", "123456789")
    account.deposit(100)
    account.withdraw(50)
    account.deposit(25)
    assert account.get_transactions() == ["Deposited: 100", "Withdrew: 50", "Deposited: 25"]
    account = BankAccount("John Doe", "123456789")
    account.deposit(100)
    account.withdraw(50)
    assert account.get_statement() == "Account Holder: John Doe\nAccount Number: 123456789\nCurrent Balance: 50\nTransactions:\n  Deposited: 100\n  Withdrew: 50\n"
    account = BankAccount("Jane Smith", "987654321")
    assert account.get_statement() == "Account Holder: Jane Smith\nAccount Number: 987654321\nCurrent Balance: 0\nTransactions:\n"

def test_check():
    check(BankAccount())
