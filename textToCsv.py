import csv


def text_to_csv(input_file, output_file):
    with open(input_file, 'r') as text_file:
        content = text_file.read().replace('\n', '')  # Read the whole content and remove newline characters

    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['input'])
        writer.writerow([content])  # Write the content as a single row in the CSV file


text_to_csv('PythonApplication1/content/meta2.txt', 'PythonApplication1/content/META.csv')