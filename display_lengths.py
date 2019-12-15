import matplotlib.pyplot as plt

with open("tweets_edited.txt", encoding="utf-8") as file:
    # line =  file.readline()
    lens = []
    for line in file:
        line_length = len(line.split())
        lens.append(line_length)
    #print(line_length)
    # print(lens)
plt.hist(x=lens, bins='auto')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.title('Tweet Length Histogram')
plt.show()
# maxfreq = n.max()
min_val = min(lens)
max_val = max(lens)
avg_val = sum(lens)/len(lens)
print(min_val, max_val, avg_val)