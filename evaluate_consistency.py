from bs4 import BeautifulSoup
from collections import Counter
import json

def calculate_consistency(modes, model):
    extracted_answers = {key: [] for key in modes}
    for mode in modes:
        # Specify the file name
        file_name = f"./results/sgp_{mode}_{model}.html"

        # Open the HTML file and read its content
        with open(file_name, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')


        # Extract and print the <h3> headings with the content "Results"
        h3_headings = soup.find_all('h3')

        # Extract and print paragraphs following <h3> headings containing "Results"
        for heading in h3_headings:
            if 'results' in heading.get_text(strip=True).lower():
                next_sibling = heading.find_next_sibling()
                while next_sibling and next_sibling.name != 'h3':
                    if next_sibling.name == 'p':
                        extracted_text = next_sibling.get_text(strip=True)
                        if 'extracted' in extracted_text.lower(): # correct, extracted
                            answer = extracted_text.split(':')[-1].strip()
                            extracted_answers[mode].append(answer)
                    next_sibling = next_sibling.find_next_sibling()

    # Get the length of the lists (assuming all lists are of equal length)
    list_length = len(next(iter(extracted_answers.values())))
    print('List length:', list_length)

    # Initialize a list to store consistency rates for each position
    consistency_rates = []

    # Iterate through each position
    for i in range(list_length):
        # Extract elements at position i from all lists
        elements_at_i = [extracted_answers[key][i] for key in extracted_answers]
        
        # Determine the majority value and its count
        counter = Counter(elements_at_i)
        majority_value, majority_count = counter.most_common(1)[0]

        # print(f"Position {i}: {majority_value} ({majority_count}/{len(extracted_answers)})")
        
        # Calculate the consistency rate at position i
        consistency_rate = majority_count / len(extracted_answers)
        
        # Append the consistency rate to the list
        consistency_rates.append(consistency_rate)

    # Calculate the average consistency rate
    average_consistency_rate = sum(consistency_rates) / list_length

    print(f"Average Consistency Rate: {average_consistency_rate:.3f}")


def calculate_average_accuracy(modes, model):
    avg_acc = 0
    for mode in modes:
        # Specify the file name
        file_name = f"./results/sgp_{mode}_{model}.json"

        with open(file_name, 'r') as file:
            data = json.load(file)

        acc = data["score"]
        avg_acc += acc
    
    avg_acc /= len(modes)
    print(f"Average accuracy: {avg_acc:.3f}")


model = 'DeepSeek-Coder-V2-16B'
modes = ['inv', 'inv_t0', 'inv_t1', 'inv_t2', 'inv_t3', 'inv_t4']
print('translation consistency for', model)
calculate_average_accuracy(modes, model)
calculate_consistency(modes, model)


modes = ['inv', 'inv_r0', 'inv_r1', 'inv_r2', 'inv_r3', 'inv_r4']
print('rotation consistency for', model)
calculate_average_accuracy(modes, model)
calculate_consistency(modes, model)
