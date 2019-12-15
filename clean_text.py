read_file = "tweets.txt"
write_file = "tweets_edited.txt"
error_count = 0

with open(read_file, encoding='UTF-8') as input_file:
    with open (write_file, 'w+',encoding='UTF-8') as output_file:
        for line in input_file:
            # line = line.lower()
            # output_file.write(line) 
            if ("https://") in line:
                # print(line)
                try:
                    pre_link, post_link = line.split("https://", 1)
                    post_link = post_link[15]
                    # print(pre_link, post_link)
                    line = pre_link + post_link
                    output_file.write(line)
                except IndexError:
                    error_count +=1
                    pass
            else:
                output_file.write(line)
print(error_count)