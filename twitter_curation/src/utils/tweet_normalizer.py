import re

def normalize(input_text:str):
    hashtags = re.findall(r'#\w+', input_text)
    mentions = re.findall(r'@\w+', input_text)
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', input_text)

    changed_text = input_text
    # for h_index in range(0, len(hashtags)):
    #     changed_text = changed_text.replace(hashtags[h_index], '<HASHTAG_' + str(h_index + 1) + '>')

    for h_index in range(0, len(mentions)):
        changed_text = changed_text.replace(mentions[h_index], '<USER_MENTION_' + str(h_index + 1) + '>')

    for h_index in range(0, len(urls)):
        changed_text = changed_text.replace(urls[h_index], '<URL_' + str(h_index + 1) + '>')

    return changed_text