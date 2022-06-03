# Copyright (c) 2020 INSTADEEP, Inc
#
# Licence INSTADEEP 2020,  AmelSellami
# LICENSE file in the root directory of this source tree.
# ======================================================================

# Goal: Convert numbers, Eastern Arabic numbers to Buckwalter, and remove
# special characters.


from lang_trans.arabic import buckwalter
from num2words import num2words

# Define the list of unique chars in The Dataset that we will remove from transcripts
List_unique_chars = [
    "’",
    "‘",
    "“",
    "ۚ",
    "à",
    "é",
    "ﺠ",
    "ﻴ",
    "ﺯ",
    "”",
    "ﺇ",
    "ﻵ",
    "İ",
    "¾",
    "﴾",
    "ﺕ",
    "ﻔ",
    "ۖ",
    "ç",
    "è",
    "ﻻ",
    "ﻑ",
    "÷",
    "ﺌ",
    "﴿",
    "ﻹ",
    "ٱ",
    "ﻷ",
    "ﻌ",
]


# Define a dictionary to convert Eastern Arabic numerals to Standard numerals
dic = {
    "٠": "0",
    "١": "1",
    "٢": "2",
    "۳": "3",
    "٤": "4",
    "۵": "5",
    "٦": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
}


def read_transcript():
    # Using readlines()
    # Open the Orthographic transcript
    transcript_file = open(
        "/path/to/transcript/folder/orthographic-transcript_old.txt", "r"
    )
    lines = transcript_file.readlines()
    return lines


def remove_unique_chars(line):
    # Remove unique characters
    for c in line:
        if c in List_unique_chars:
            line = line.replace(c, "")
    return line


def split_line(line):
    # Strips the newline character
    list_line = line.split()
    return list_line


def has_numbers(input_string):
    st = input_string.split("</s>")
    st = st[0]
    return any(char.isdigit() for char in st)


def convert_arabic_numerals_tostandard_numerl(line, dic):
    # Convert Arabic numerals to English numerals
    for k in dic:
        pos = line.find(k)
        line.replace(line[pos], dic[k])
    return line


def convert_numb_to_words(list_line, count):
    # Convert Numbers to backwalter words
    for word in list_line[:-1]:
        number = "".join(filter(lambda i: i.isdigit(), word))
        if len(number) > 0:
            pos = word.find(number[0])
            numb_to_word = num2words(int(number), lang="ar")
            arabic_to_backwalter = buckwalter.transliterate(numb_to_word)
            list_line[count] = (
                word[:pos] + arabic_to_backwalter + word[pos + len(number) :]
            )
        count = count + 1
    new_line = " ".join(list_line)

    return new_line


def write_to_new_transcript(new_line):
    # Writing to a new script
    f = open("/path/to/transcript/folder/orthographic-transcript.txt", "a")
    f.write(new_line)
    f.write("\n")


def main():
    # Read the lines
    lines = read_transcript()
    # Loop over the lines
    for line in lines:
        count = 0
        # remove lines containing english words
        if "@" in line:
            continue
        # Remove unique chars defined above
        line = remove_unique_chars(line)
        # Convert the Arabic numeral to standard numerals
        line = convert_arabic_numerals_tostandard_numerl(line, dic)
        # Split the lines
        list_line = split_line(line)
        # Convert numerals to words
        new_line = convert_numb_to_words(list_line, count)
        # Write to a new trasncript if line has no number in it
        if not has_numbers(new_line):
            write_to_new_transcript(new_line)


if __name__ == "__main__":
    # Execute main
    main()
