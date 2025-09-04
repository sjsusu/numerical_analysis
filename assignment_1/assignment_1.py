import math
from decimal import *

"""
MATH 5334 - Assignment 1: Binary Encoding
Name: Serena Su

INSTRUCTIONS:
Follow the notes on Canvas and write a program that converts a decimal number string into IEEE-754 64 bit binary.

Please implement the conversion yourself without calling any built-in functions.
Required submission files:
1. A code documentation in pdf format showing your algorithm.
2. A source code implemented in either Matlab or Python.

For your program, it should take an input: Decimal number, and then output the 64 binary.
Corner cases you may consider: 
1. 0s (both +0 and -0)
2. Positive and Negative values?
3. Integer vs fraction number?
4. Decimal values falls exactly between two 64bits?
and more...
"""


def decimal_to_binary(num_string):
    """
    This function takes a decimal (string) and returns the number in IEEE 754 64-bit binary form (string). The returned string has spaces to indicate the sign, exponent, and fraction.

    Parameters
    ----------
    num_string: string
        The decimal provided from user input.

    Returns
    -------
    string:
        num_string converted to IEEE 754 64-bit binary. Formatted as "sign exponent fraction".

    Raises
    ------
    TypeError
        The input is not a decimal.
    ValueError
        The input is too small or large for IEEE 754 64-bit conversion.
    """

    # Check if user input is a decimal number
    try:
        num = Decimal(num_string)
        num_abs = num.copy_abs()
    except InvalidOperation:
        raise TypeError("Input is not a decimal number, try again.")

    # Assign the sign by looking for '-' from original input.
    sign = "1" if num_string.strip()[0] == "-" else "0"

    # Check cases for different types of floats
    # 1) Infinity (+ or -)
    if math.isinf(num):
        exponent = "".join(["1" for i in range(11)])
        fraction = "".join(["0" for i in range(52)])
        return " ".join([sign, exponent, fraction])

    # 2) NaN
    elif math.isnan(num):
        exponent = "".join(["1" for i in range(11)])
        fraction = "1" + "".join(["0" for i in range(51)])
        return " ".join([sign, exponent, fraction])

    # 3) Zero (+ or -)
    elif num_abs == Decimal("0"):
        exponent = "".join(["0" for i in range(11)])
        fraction = "".join(["0" for i in range(52)])
        return " ".join([sign, exponent, fraction])

    # Check Magnitude
    # Too small
    elif num_abs < Decimal('1e-308'):
        raise ValueError("Input is too small, try again.")
    # Too large
    elif num_abs > Decimal('1e308'):
        raise ValueError("Input is too large, try again.")

    # 4) General Case
    else:
        # Retrieve integer and fractional parts of number
        integer = num_abs.quantize(Decimal("1"), rounding=ROUND_FLOOR)
        fraction = num_abs - integer

        # Convert integer portion to binary
        integer_binary = integer_to_binary(float(integer))

        # Convert fraction to binary and update integer binary if rounded
        fraction_binary, integer_binary = fraction_to_binary(fraction, integer_binary)

        # Find exponent
        if len(integer_binary) != 0:
            exponent = len(integer_binary) - 1
        else:
            exponent = -1 * (fraction_binary.index("1") + 1)

        # Find biased exponent and binary representation
        exponent_biased = exponent + 1023
        exponent_binary = integer_to_binary(exponent_biased)

        # Append extra zeros to exponent
        if len(exponent_binary) != 11:
            missing_bits = 11 - len(exponent_binary)
            exponent_binary = ["0" for i in range(missing_bits)] + exponent_binary

        # Combine integer and fraction binary
        integer_and_fraction_binary = integer_binary + fraction_binary

        # Normalization
        first_one = integer_and_fraction_binary.index("1")
        mantissa = integer_and_fraction_binary[(first_one + 1) :]

        # Build final binary number
        num_binary = [sign, " "] + exponent_binary + [" "] + mantissa

        return "".join(num_binary)


# Helper functions
def integer_to_binary(integer):
    """
    This function takes an integer and returns its binary representation as a list of chars.

    Returns empty list if integer entered is any form of 0.

    Parameters
    ----------
    integer: int or float
        The integer to be converted into binary.

    Returns
    -------
    list: string
        The binary representation of integer as ordered list. Each index contains a '0' or '1'.
    """
    dividend = integer
    integer_binary = []
    while True:
        quotient = dividend / 2
        remainder = math.floor(dividend % 2)
        integer_binary.append(str(remainder))

        if quotient < 1:
            break
        else:
            dividend = quotient

    # Remove trailing zeros
    while integer_binary and integer_binary[-1] == "0":
        integer_binary.pop()

    # Return reversed binary
    return integer_binary[::-1]


def fraction_to_binary(fraction, integer_binary):
    """
    This function takes a fractional portion of a number (as a decimal) and returns a tuple of
    (1) its binary representation as a list of chars and
    (2) integer_binary (copy or rounded).

    The length of list (1) is based on the number of bits remaining in the final mantissa.

    Parameters
    ----------
    fraction: Decimal object
        Fraction portion of number to be converted into binary.
    integer_binary: list of string
        The binary representation of integer portion of number as list of chars '0' and '1'.
        Used to calculated remaining bits (significant figures) in mantissa (before normalization).

    Returns
    -------
    tuple of (list of string, list of string)
        -  1st list of string:

            The binary representation of fraction as ordered list. Each index contains a '0' or '1'.

        - 2nd list of string:

            If rounding occurs in the integers, a rounded version of integer_binary is provided.
            Otherwise, a copy of the original integer_binary is given.
    """

    TOLERANCE = Decimal("1e-14")
    MAX_SIG_FIGS = 54

    # Calculates current significant figures
    sig_figs = len(integer_binary)
    fraction_binary = []
    dividend = fraction

    # Convert fraction to binary
    while sig_figs < MAX_SIG_FIGS:
        dividend *= Decimal("2")
        remainder = dividend % Decimal("2")
        remainder = remainder.quantize(Decimal("1"), rounding=ROUND_FLOOR)
        dividend -= remainder
        fraction_binary.append(str(remainder))

        if dividend <= TOLERANCE:
            break
        elif (sig_figs == 0 and remainder == 1) or (sig_figs != 0):
            sig_figs += 1

    # Non-Rounding Cases
    # 1) Early Truncation
    if sig_figs <= (MAX_SIG_FIGS - 1):
        # Adds extra zeros if truncated early
        fraction_binary += (53 - sig_figs) * ["0"]
        return (fraction_binary, integer_binary)

    # 2) Rounding is not needed (last digit is 0)
    elif fraction_binary[len(fraction_binary) - 1] == "0":
        fraction_binary.pop()
        return (fraction_binary, integer_binary)

    # Rounding Case (last digit is 1)
    else:
        fraction_binary.pop()
        fraction_binary, rounding_integer = round_up(fraction_binary)
        if rounding_integer:
            integer_binary_rounded, appending_integer = round_up(integer_binary)

            if appending_integer:
                integer_binary_rounded.insert(0, "1")

            integer_binary = integer_binary_rounded

        return (fraction_binary, integer_binary)


def round_up(binary):
    """
    Built as a helper to fraction_to_binary().

    ASSUMES that a binary number will be rounded up. This function rounds up the given binary number starting with the last number WITHOUT PREPENDING.

    Returns a tuple of:
    - rounded binary number
    - boolean indicating if '1' will need to appended to beginning.

    Parameters
    ----------
    binary: list of string
        Binary representation of a number as ordered list of chars '0' and '1'.

        THE FOLLOWING ASSUMPTIONS ARE MADE:

        If the number is a(n):
            - Integer:

                '1' to be rounded remains in the binary representation as the last digit.

            - Fraction:

                The last digit '1' to be rounded is removed from binary representation.

    Returns
    -------
    tuple of (list of string, bool)
        - list of string:

            The rounded binary number as an ordered list of chars '0' and '1'

        - bool:

            Indicates if a '1' needs to be appended to the beginning of from rounding.
    """

    length = len(binary)

    # Case if integer is 0
    if length == 0:
        return (binary, True)

    # General Case
    for i in range(length - 1, -1, -1):
        if binary[i] == "0":
            binary[i] = "1"
            return (binary, False)
        elif binary[i] == "1":
            binary[i] = "0"

        if i == 0:
            return (binary, True)


# Main method
if __name__ == "__main__":
    line = "".join(["-" for i in range(74)])
    print(line, "Assignment 1: Binary Encoding", line, sep="\n")

    while True:
        print("Note: Enter q to quit.", sep="\n")
        num = input("Enter a number(in decimal form): ")

        # Quit condition
        if num.lower() == "q":
            break

        try:
            binary = decimal_to_binary(num)
            print(f"Binary: {binary}\n")
        except Exception as e:
            print(e, "\n")

    print(line)
