from typing import *

class Car:
    def __init__(self, make, model, year, mileage, color):
        """
        Initializes a new instance of the Car class.

        :param make: The make of the car.
        :param model: The model of the car.
        :param year: The year the car was manufactured.
        :param mileage: The mileage of the car in miles.
        :param color: The color of the car.
        """
        self.make = make
        self.model = model
        self.year = year
        self.mileage = mileage
        self.color = color

    def __str__(self):
        """
        Returns a string representation of the car.

        :return: A string in the format: Car(make='make', model='model', year=year, mileage=mileage, color='color')
        """
        return (f"Car(make='{self.make}', model='{self.model}', year={self.year}, "
                f"mileage={self.mileage}, color='{self.color}')")

    def drive(self, distance):
        """
        Increases the car's mileage by the specified distance.

        :param distance: The distance to drive.
        :raises ValueError: If the distance is negative.
        """
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        self.mileage += distance

    def paint(self, new_color):
        """
        Changes the car's color to the specified new color.

        :param new_color: The new color of the car.
        """
        self.color = new_color

### Unit tests below ###
def check(candidate):
    assert str(Car("Toyota", "Corolla", 2020, 15000, "blue")) == "Car(make='Toyota', model='Corolla', year=2020, mileage=15000, color='blue')"
    assert Car("Honda", "Civic", 2018, 20000, "red").make == "Honda"
    assert Car("Ford", "Mustang", 2021, 5000, "black").model == "Mustang"
    assert Car("Chevrolet", "Impala", 2019, 12000, "white").year == 2019
    assert Car("Tesla", "Model S", 2022, 0, "silver").mileage == 0
    assert Car("BMW", "X5", 2017, 30000, "gray").color == "gray"
    car = Car("Audi", "A4", 2016, 18000, "green")
    car.drive(500)
    assert car.mileage == 18500
    car = Car("Mercedes", "C-Class", 2015, 25000, "brown")
    car.paint("yellow")
    assert car.color == "yellow"
    car = Car("Lamborghini", "Aventador", 2023, 1000, "orange")
    car.drive(0)
    assert car.mileage == 1000
    car = Car("Porsche", "911", 2020, 7000, "blue")
    try:
    car.drive(-100)
    except ValueError as e:
    assert str(e) == "Distance cannot be negative"

def test_check():
    check(Car())
