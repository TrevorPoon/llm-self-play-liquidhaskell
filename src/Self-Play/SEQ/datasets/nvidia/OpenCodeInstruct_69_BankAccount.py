from typing import *

class BankAccount:
    def __init__(self):
        """
        Initializes a new bank account with a balance of 0.
        """
        self.balance = 0

    def deposit(self, amount):
        """
        Deposits a specified amount into the account if the amount is positive.
        
        :param amount: The amount to deposit.
        """
        if amount > 0:
            self.balance += amount
        else:
            print("Deposit amount must be positive.")

    def withdraw(self, amount):
        """
        Withdraws a specified amount from the account if the amount is positive and sufficient funds are available.
        
        :param amount: The amount to withdraw.
        """
        if amount > self.balance:
            print("Insufficient funds.")
        elif amount < 0:
            print("Withdrawal amount must be positive.")
        else:
            self.balance -= amount

    def check_balance(self):
        """
        Returns the current balance of the account.
        
        :return: The current balance.
        """
        return self.balance

### Unit tests below ###
def check(candidate):
    assert candidate.check_balance() == 0
    assert candidate.deposit(100) is None
    assert candidate.withdraw(100) is None
    account = candidate; account.deposit(100); assert account.check_balance() == 100
    account = candidate; account.deposit(100); account.withdraw(50); assert account.check_balance() == 50
    account = candidate; account.deposit(100); account.withdraw(150); assert account.check_balance() == 100
    account = candidate; account.deposit(-100); assert account.check_balance() == 0
    account = candidate; account.withdraw(-100); assert account.check_balance() == 0
    account = candidate; account.deposit(100); account.deposit(50); assert account.check_balance() == 150
    account = candidate; account.deposit(100); account.withdraw(100); account.withdraw(100); assert account.check_balance() == 0

def test_check():
    check(BankAccount())
