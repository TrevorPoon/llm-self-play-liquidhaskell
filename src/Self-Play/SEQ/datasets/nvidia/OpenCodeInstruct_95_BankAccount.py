from typing import *

class BankAccount:
    def __init__(self, initial_balance=0):
        """
        Initialize a bank account with an optional initial balance.
        
        :param initial_balance: The starting balance of the account (default is 0).
        """
        self.balance = initial_balance

    def deposit(self, amount):
        """
        Deposit a specified amount of money into the account.
        
        :param amount: The amount to deposit (must be positive).
        :raises ValueError: If the deposit amount is not positive.
        """
        if amount > 0:
            self.balance += amount
        else:
            raise ValueError("Deposit amount must be positive.")
    
    def withdraw(self, amount):
        """
        Withdraw a specified amount of money from the account.
        
        :param amount: The amount to withdraw (must be positive).
        :raises ValueError: If the withdrawal amount is not positive or if there is insufficient balance.
        """
        if amount > 0:
            if self.balance >= amount:
                self.balance -= amount
            else:
                raise ValueError("Insufficient balance for the withdrawal.")
        else:
            raise ValueError("Withdrawal amount must be positive.")
    
    def check_balance(self):
        """
        Check the current balance of the account.
        
        :return: The current balance of the account.
        """
        return self.balance

### Unit tests below ###
def check(candidate):
    assert candidate.check_balance() == 0
    assert BankAccount(100).check_balance() == 100
    assert BankAccount(50).deposit(50) == None and BankAccount(50).check_balance() == 100
    assert BankAccount(100).withdraw(50) == None and BankAccount(100).check_balance() == 50
    assert BankAccount(100).withdraw(100) == None and BankAccount(100).check_balance() == 0
    assert BankAccount(100).withdraw(150) == None and BankAccount(100).check_balance() == 100
    except ValueError as e:
    assert str(e) == "Insufficient balance for the withdrawal."
    assert candidate.deposit(100) == None and candidate.check_balance() == 100
    assert candidate.deposit(-100) == None and candidate.check_balance() == 0
    except ValueError as e:
    assert str(e) == "Deposit amount must be positive."
    assert candidate.withdraw(-100) == None and candidate.check_balance() == 0
    except ValueError as e:
    assert str(e) == "Withdrawal amount must be positive."
    account = BankAccount(100)
    account.deposit(50)
    account.withdraw(30)
    assert account.check_balance() == 120

def test_check():
    check(BankAccount())
