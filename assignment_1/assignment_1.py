import math

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
    This function takes a decimal (string) and returns the number in 64-bit binary form (string). The returned string has spaces to indicate the sign, exponent, and fraction.

    Parameter:
        num_string (string) : user input

    Return:
        string : num_string in IEEE 754 64-bit binary. Formatted as "sign exponent fraction".
    """

    # Checks if user input is a valid decimal number within IEEE-754 64 bit binary.
    try:
        num = float(num_string)

    # Not a valid decimal input
    except ValueError:
        raise ValueError("Input is not a decimal number, try again.")

    # Overflow
    except OverflowError:
        raise OverflowError("Overflow error. Input is too large, try again.")

    # TODO: underflow?

    # Assigns the sign by looking for '-' from original input.
    sign = "1" if num_string.strip()[0] == "-" else "0"

    # Checks Cases for different types of floats
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
    elif num == 0:
        exponent = "".join(["0" for i in range(11)])
        fraction = "".join(["0" for i in range(52)])
        return " ".join([sign, exponent, fraction])

    # 4) General Case
    else:
        # Retrieve integer and fractional parts of number
        fraction, integer = math.modf(num)

        # Convert integer portion to binary
        integer_binary = integer_to_binary(integer)

        # Convert fraction to binary
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
    This function takes and integer and returns its binary representation as a list of chars.
    Returns empty list if integer entered is any form of 0.
    Parameters:
        integer: int
    Returns:
        list (string): The binary representation of integer as ordered list. Each index contains a '0' or '1'.
    """
    dividend = math.fabs(integer)
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
    (2) integer_binary (copy or rounded)

    The length of list (1) is based on the number of bits remaining in the final mantissa.
    Parameters:
        fraction: float
        integer_binary: list(string):
            The binary representation of integer portion of number as list of chars '0' and '1'.
            Used to calculated remaining bits (significant figures) in mantissa (before normalization).
    Returns:
        tuple: (fraction_binary, integer_binary_rounded)

            fraction_binary:
            list(string): The binary representation of fraction as ordered list. Each index contains a '0' or '1'.

            integer_binary_rounded:
            list(string): If rounding occurs in the integers, a rounded version of integer_binary is provided. Otherwise, a copy of the original integer_binary is given.
    """

    TOLERANCE = 1e-14

    # Calculates current significant figures
    sig_figs = len(integer_binary)
    fraction_binary = []
    dividend = math.fabs(fraction)

    # Convert fraction to binary
    while sig_figs < 54:
        dividend *= 2
        remainder = math.floor(dividend) % 2
        dividend -= remainder

        if dividend <= TOLERANCE:
            break
        elif (sig_figs == 0 and remainder == 1) or (sig_figs != 0):
            sig_figs += 1

        fraction_binary.append(str(remainder))

    # Non-Rounding Cases
    if sig_figs <= 53:
        # Adds extra zeros if truncated early
        fraction_binary += (53 - sig_figs) * ["0"]
        return (fraction_binary, integer_binary)

    elif fraction_binary[53] == "0":
        fraction_binary.pop()
        return (fraction_binary, integer_binary)

    # Rounding Case
    else:
        fraction_binary, rounding_integers = round_up(fraction_binary)
        if rounding_integers:
            integer_binary_rounded, appending_integer = round_up(integer_binary)

            if appending_integer:
                integer_binary_rounded.insert(0, "1")

        return (fraction_binary, integer_binary_rounded)


def round_up(binary):
    """
    Built as a helper to fraction_to_binary().

    ASSUMES that a binary number will be rounded up. This function rounds up the given binary number starting with the last number WITHOUT PREPENDING. Returns a tuple of (1) rounded binary number (2) boolean indicating if '1' will need to appended to beginning.
    Parameters:
        binary: list(string): Binary representation of a number as ordered list of chars '0' and '1'.

            THE FOLLOWING ASSUMPTIONS ARE MADE:
            If the number is a(n):
                (i) INTEGER: '1' to be rounded remains in the binary representation as the last digit.
                (ii) FRACTION: The last digit '1' to be rounded is removed from binary representation.
    Returns:
        tuple: (rounded_binary, prepend)

        rounded_binary: list(string): The rounded binary number as an ordered list of chars '0' and '1'.

        prepend: bool: Indicates if a '1' needs to be appended to the beginning of rounded_binary.
    """

    length = len(binary)

    if length == 0:
        return (binary, True)

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
    print(
        "Remark: Tail end of mantissa may not be fully \naccurate due to machine error.",
        line,
        sep="\n",
    )

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
