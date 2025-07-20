from typing import *

def update_email(user_profiles, username, new_email):
    """
    Updates the email field for a specified user in the user_profiles dictionary.

    Parameters:
    user_profiles (dict): A dictionary where keys are usernames and values are dictionaries containing user details.
    username (str): The username of the user whose email needs to be updated.
    new_email (str): The new email address to be set for the user.

    Raises:
    KeyError: If the specified username does not exist in the user_profiles dictionary.
    """
    if username in user_profiles:
        user_profiles[username]['email'] = new_email
    else:
        raise KeyError(f"Username '{username}' not found in user profiles.")

### Unit tests below ###
def check(candidate):
    assert candidate({"alice": {"email": "alice@example.com", "age": 30, "city": "New York"}}, "alice", "alice_new@example.com") is None
    assert candidate({"bob": {"email": "bob@example.com", "age": 25, "city": "Los Angeles"}}, "bob", "bob_new@example.com") is None
    assert candidate({"charlie": {"email": "charlie@example.com", "age": 35, "city": "Chicago"}}, "charlie", "charlie_new@example.com") is None
    assert candidate({"dave": {"email": "dave@example.com", "age": 40, "city": "Houston"}}, "dave", "dave_new@example.com") is None
    try:
    candidate({"eve": {"email": "eve@example.com", "age": 28, "city": "Phoenix"}}, "frank", "frank_new@example.com")
    except KeyError as e:
    assert str(e) == "Username 'frank' not found in user profiles."
    try:
    candidate({}, "grace", "grace_new@example.com")
    except KeyError as e:
    assert str(e) == "Username 'grace' not found in user profiles."
    assert candidate({"heidi": {"email": "heidi@example.com", "age": 22, "city": "Philadelphia"}}, "heidi", "heidi_new@example.com") is None
    try:
    candidate({"ivan": {"email": "ivan@example.com", "age": 33, "city": "San Antonio"}}, "judy", "judy_new@example.com")
    except KeyError as e:
    assert str(e) == "Username 'judy' not found in user profiles."
    assert candidate({"karen": {"email": "karen@example.com", "age": 29, "city": "San Diego"}}, "karen", "karen_new@example.com") is None
    try:
    candidate({"leo": {"email": "leo@example.com", "age": 45, "city": "Dallas"}}, "mike", "mike_new@example.com")
    except KeyError as e:
    assert str(e) == "Username 'mike' not found in user profiles."

def test_check():
    check(update_email)
