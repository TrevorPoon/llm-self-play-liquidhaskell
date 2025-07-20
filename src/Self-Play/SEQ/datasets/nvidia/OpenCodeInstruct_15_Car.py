from typing import *

class Car:
    def __init__(self, make, model, year):
        """
        Initializes a new Car instance with the specified make, model, and year.

        :param make: A string representing the manufacturer of the car.
        :param model: A string representing the model of the car.
        :param year: An integer representing the year the car was manufactured.
        """
        self.make = make
        self.model = model
        self.year = year

    def description(self):
        """
        Returns a formatted string with the car's information.

        :return: A string in the format "make model (year)".
        """
        return f"{self.make} {self.model} ({self.year})"

### Unit tests below ###
def check(candidate):
    assert Car("Toyota", "Camry", 2021).description() == "Toyota Camry (2021)"
    assert Car("Honda", "Civic", 2020).description() == "Honda Civic (2020)"
    assert Car("Ford", "Mustang", 1969).description() == "Ford Mustang (1969)"
    assert Car("Chevrolet", "Corvette", 2023).description() == "Chevrolet Corvette (2023)"
    assert Car("Tesla", "Model S", 2022).description() == "Tesla Model S (2022)"
    assert Car("BMW", "M3", 2018).description() == "BMW M3 (2018)"
    assert Car("Audi", "R8", 2019).description() == "Audi R8 (2019)"
    assert Car("Lamborghini", "Aventador", 2021).description() == "Lamborghini Aventador (2021)"
    assert Car("Porsche", "911", 2020).description() == "Porsche 911 (2020)"
    assert Car("Ferrari", "488", 2019).description() == "Ferrari 488 (2019)"

def test_check():
    check(Car())
