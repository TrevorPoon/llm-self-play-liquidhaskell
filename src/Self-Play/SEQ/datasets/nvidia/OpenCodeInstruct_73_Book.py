from typing import *

class Book:
    def __init__(self, title: str, author: str, year_published: int):
        """
        Initialize a new Book instance.

        :param title: The title of the book.
        :param author: The author of the book.
        :param year_published: The year the book was published.
        """
        self.title = title
        self.author = author
        self.year_published = year_published

    def get_book_info(self) -> str:
        """
        Return a formatted string containing the book's title, author, and year of publication.

        :return: A string in the format "Title: <title>, Author: <author>, Year: <year_published>".
        """
        return f"Title: {self.title}, Author: {self.author}, Year: {self.year_published}"

### Unit tests below ###
def check(candidate):
    assert Book("1984", "George Orwell", 1949).get_book_info() == "Title: 1984, Author: George Orwell, Year: 1949"
    assert Book("To Kill a Mockingbird", "Harper Lee", 1960).get_book_info() == "Title: To Kill a Mockingbird, Author: Harper Lee, Year: 1960"
    assert Book("The Great Gatsby", "F. Scott Fitzgerald", 1925).get_book_info() == "Title: The Great Gatsby, Author: F. Scott Fitzgerald, Year: 1925"
    assert Book("Pride and Prejudice", "Jane Austen", 1813).get_book_info() == "Title: Pride and Prejudice, Author: Jane Austen, Year: 1813"
    assert Book("Moby Dick", "Herman Melville", 1851).get_book_info() == "Title: Moby Dick, Author: Herman Melville, Year: 1851"
    assert Book("War and Peace", "Leo Tolstoy", 1869).get_book_info() == "Title: War and Peace, Author: Leo Tolstoy, Year: 1869"
    assert Book("The Catcher in the Rye", "J.D. Salinger", 1951).get_book_info() == "Title: The Catcher in the Rye, Author: J.D. Salinger, Year: 1951"
    assert Book("The Hobbit", "J.R.R. Tolkien", 1937).get_book_info() == "Title: The Hobbit, Author: J.R.R. Tolkien, Year: 1937"
    assert Book("Brave New World", "Aldous Huxley", 1932).get_book_info() == "Title: Brave New World, Author: Aldous Huxley, Year: 1932"
    assert Book("Fahrenheit 451", "Ray Bradbury", 1953).get_book_info() == "Title: Fahrenheit 451, Author: Ray Bradbury, Year: 1953"

def test_check():
    check(Book())
