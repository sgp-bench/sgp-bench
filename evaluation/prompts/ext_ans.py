

# pids = 852,  104,  824,  506,  540

demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.
Question: What is the primary color of the object in the image?\n\nA) Red\nB) Blue\nC) Black\nD) Green

Model response: **Step 1: Examine the image**\n\nThe image consists of various shapes filled with different colors. We need to identify the primary color of the object in the image.\n\n**Step 2: Focus on the dominant color**\n\nThe shapes that cover the largest area in the object are filled with shades of blue and its variations.\n\n**Answer: B**

Extracted answer: B

Hint: The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.
Question: What is the background color of the image?\n\nA) Red\nB) Green\nC) Blue\nD) Yellow

Model response: Answer: The background color of the code is blue.

Extracted answer: C

Hint: The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.
Question: What is the shape of the buckle on the object?\n\nA) Circle\nB) Triangle\nC) Square\nD) Hexagon

Model response: Answer: D) Hexagon.

Extracted answer: D

Hint: The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.
Question: What type of object is shown in the image?\n\nA) Watch\nB) Belt\nC) Bracelet\nD) Necklace

Model response: The object in the code is a watch.

Extracted answer: A

Hint: The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.
Question: What is the primary color of the object in the image?\n\nA) Blue\nB) Yellow\nC) Green\nD) Red

Model response: The primary color of the object in the code is yellow.

Extracted answer: B
"""