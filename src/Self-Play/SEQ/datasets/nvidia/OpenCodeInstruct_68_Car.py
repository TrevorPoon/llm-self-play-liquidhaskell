from typing import *

class Car:
    def __init__(self, make, model, year, color):
        """
        Initialize a new Car instance.

        :param make: The make of the car.
        :param model: The model of the car.
        :param year: The year the car was manufactured.
        :param color: The color of the car.
        """
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.speed = 0  # Initialize the speed attribute to 0

    def start_engine(self):
        """
        Start the car's engine.
        """
        print("Engine started.")

    def stop_engine(self):
        """
        Stop the car's engine.
        """
        print("Engine stopped.")

    def paint(self, new_color):
        """
        Paint the car a new color.

        :param new_color: The new color for the car.
        """
        self.color = new_color
        print(f"The car is now {self.color}.")

    def accelerate(self, speed_increase):
        """
        Increase the car's speed.

        :param speed_increase: The amount to increase the car's speed by.
        """
        self.speed += speed_increase
        print(f"The car has accelerated by {speed_increase} mph. Current speed: {self.speed} mph.")

    def brake(self, speed_decrease):
        """
        Decrease the car's speed.

        :param speed_decrease: The amount to decrease the car's speed by.
        """
        if speed_decrease > self.speed:
            self.speed = 0
            print("The car has come to a complete stop.")
        else:
            self.speed -= speed_decrease
            print(f"The car has slowed down by {speed_decrease} mph. Current speed: {self.speed} mph.")

    def __str__(self):
        """
        Return a string representation of the car.

        :return: A string describing the car.
        """
        return f"{self.year} {self.make} {self.model} in {self.color} with current speed of {self.speed} mph."

### Unit tests below ###
def check(candidate):
    assert Car("Toyota", "Corolla", 2021, "red").make == "Toyota"
    assert Car("Toyota", "Corolla", 2021, "red").model == "Corolla"
    assert Car("Toyota", "Corolla", 2021, "red").year == 2021
    assert Car("Toyota", "Corolla", 2021, "red").color == "red"
    assert Car("Toyota", "Corolla", 2021, "red").speed == 0
    car = Car("Honda", "Civic", 2020, "blue")
    car.accelerate(50)
    assert car.speed == 50
    car = Car("Honda", "Civic", 2020, "blue")
    car.accelerate(50)
    car.brake(20)
    assert car.speed == 30
    car = Car("Honda", "Civic", 2020, "blue")
    car.accelerate(50)
    car.brake(60)
    assert car.speed == 0
    car = Car("Ford", "Mustang", 2019, "black")
    car.paint("red")
    assert car.color == "red"
    assert str(Car("Tesla", "Model S", 2022, "white")) == "2022 Tesla Model S in white with current speed of 0 mph."

def test_check():
    check(Car())
