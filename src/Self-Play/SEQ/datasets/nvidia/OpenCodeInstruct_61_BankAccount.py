from typing import *

class BankAccount:
    def __init__(self, initial_balance):
        """
        Initializes the bank account with a given initial balance.
        
        :param initial_balance: The starting balance of the account.
        """
        self.balance = initial_balance
    
    def deposit(self, amount):
        """
        Deposits a specified amount of money into the account if the amount is positive.
        
        :param amount: The amount to deposit.
        """
        if amount > 0:
            self.balance += amount
    
    def withdraw(self, amount):
        """
        Withdraws a specified amount of money from the account if the amount is positive and sufficient funds are available.
        
        :param amount: The amount to withdraw.
        """
        if 0 < amount <= self.balance:
            self.balance -= amount
    
    def get_balance(self):
        """
        Returns the current balance of the account.
        
        :return: The current balance.
        """
        return self.balance

### Unit tests below ###
def check(candidate):
    assert BankAccount(100).get_balance() == 100
    assert BankAccount(0).get_balance() == 0
    assert BankAccount(-50).get_balance() == -50
    assert BankAccount(100).deposit(50) or BankAccount(100).get_balance() == 150
    assert BankAccount(100).deposit(-50) or BankAccount(100).get_balance() == 100
    assert BankAccount(100).withdraw(50) or BankAccount(100).get_balance() == 50
    assert BankAccount(100).withdraw(150) or BankAccount(100).get_balance() == 100
    assert BankAccount(100).withdraw(-50) or BankAccount(100).get_balance() == 100
    assert BankAccount(100).deposit(50) or BankAccount(100).withdraw(30) or BankAccount(100).get_balance() == 120
    assert BankAccount(100).withdraw(100) or BankAccount(100).get_balance() == 0

def test_check():
    check(BankAccount())
