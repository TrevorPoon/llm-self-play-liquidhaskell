from typing import *

class Book:
    def __init__(self, title, author, pages):
        """
        Initializes a new Book instance with the given title, author, and number of pages.

        :param title: The title of the book (string).
        :param author: The author of the book (string).
        :param pages: The number of pages in the book (integer).
        """
        self.title = title
        self.author = author
        self.pages = pages

    def describe_book(self):
        """
        Returns a string describing the book in the format:
        "Title: <title>, Author: <author>, Pages: <pages>".

        :return: A formatted string describing the book.
        """
        return f"Title: {self.title}, Author: {self.author}, Pages: {self.pages}"

### Unit tests below ###
def check(candidate):
    assert Book("The Great Gatsby", "F. Scott Fitzgerald", 180).describe_book() == "Title: The Great Gatsby, Author: F. Scott Fitzgerald, Pages: 180"
    assert Book("1984", "George Orwell", 328).describe_book() == "Title: 1984, Author: George Orwell, Pages: 328"
    assert Book("To Kill a Mockingbird", "Harper Lee", 281).describe_book() == "Title: To Kill a Mockingbird, Author: Harper Lee, Pages: 281"
    assert Book("Pride and Prejudice", "Jane Austen", 432).describe_book() == "Title: Pride and Prejudice, Author: Jane Austen, Pages: 432"
    assert Book("The Catcher in the Rye", "J.D. Salinger", 277).describe_book() == "Title: The Catcher in the Rye, Author: J.D. Salinger, Pages: 277"
    assert Book("", "", 0).describe_book() == "Title: , Author: , Pages: 0"
    assert Book("Short Story", "Author Name", 10).describe_book() == "Title: Short Story, Author: Author Name, Pages: 10"
    assert Book("A Tale of Two Cities", "Charles Dickens", 328).describe_book() == "Title: A Tale of Two Cities, Author: Charles Dickens, Pages: 328"
    assert Book("War and Peace", "Leo Tolstoy", 1225).describe_book() == "Title: War and Peace, Author: Leo Tolstoy, Pages: 1225"
    assert Book("Moby Dick", "Herman Melville", 635).describe_book() == "Title: Moby Dick, Author: Herman Melville, Pages: 635"

def test_check():
    check(Book())
