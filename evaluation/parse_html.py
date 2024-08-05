from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm


results_dir = os.path.join(os.getcwd(), '..', 'results')

for file in os.listdir(results_dir):
    if file.endswith('.html') and 'sgp_svg' in file:
        result_html = os.path.join(results_dir, file)
    else:
        continue
    
    result_json = result_html.replace('.html', '_query.json')
    if os.path.exists(result_json):
        print(f"JSON file {result_json} already exists. Skipping ...")
        continue

    print(f"Processing {file} ...")

    # Load HTML content from a file
    with open(result_html, 'r') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all <h3> tags, assuming each represents a new section
    h3_tags = soup.find_all('h3')
    all_entries = []

    # Iterate through each <h3> tag
    for h3 in tqdm(h3_tags):
        entry = {}
        section_name = h3.get_text().strip().lower()  # Normalize the header name
        content = []
        current_element = h3.next_sibling

        # Iterate through siblings until the next <h3> tag
        while current_element and current_element.name != 'h3':
            if current_element.name:  # Checks if it's a tag and not a NavigableString
                content.append(current_element.get_text(strip=True))
            current_element = current_element.next_sibling

        # Store the joined content under the appropriate key
        if content:  # Only add if content is not empty
            entry[section_name] = ' '.join(content)
            if 'prompt conversation' in entry or 'sampled message' in entry or 'results' in entry:
                # import ipdb; ipdb.set_trace()
                if section_name == 'prompt conversation':
                    entry[section_name] = entry[section_name].split('Question:')[-1].split('\nImportant')[0].strip()
                elif section_name == 'sampled message':
                    entry[section_name] = entry[section_name].split('assistant')[-1].strip()
                elif section_name == 'results':
                    entry[section_name] = entry[section_name].split('Extracted Answer:')[0].strip()
                else:
                    raise ValueError(f"Unknown section name: {section_name}")

                all_entries.append(entry)

    # Convert the list of entries to JSON and save it to a file
    with open(result_json, 'w') as json_file:
        json.dump(all_entries, json_file, indent=4)

    print(f"JSON file {result_json} with {len(all_entries)} has been saved.")