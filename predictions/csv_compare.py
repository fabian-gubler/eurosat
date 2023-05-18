import csv

def compare_csv_files(file1, file2):
    count_differing_entries = 0
    count_total_entries = 0

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        csv_reader1 = csv.reader(f1)
        csv_reader2 = csv.reader(f2)

        for row1, row2 in zip(csv_reader1, csv_reader2):
            for col1, col2 in zip(row1, row2):
                if col1 != col2:
                    count_differing_entries += 1
                count_total_entries += 1

    percentage_diff = (count_differing_entries / count_total_entries) * 100

    return percentage_diff

# Usage example
csv_file1 = 'predictions_transfer_1.csv'
csv_file2 = 'resnet50_selflearning_517_test_5l_aug.score_58.csv'

percentage_diff = compare_csv_files(csv_file1, csv_file2)
print(f"Percentage of differing entries: {percentage_diff:.2f}%")
