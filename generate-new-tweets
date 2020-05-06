genimport gpt_2_simple as gpt2
import os
import requests
import csv

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/124M/

file_name = "elonmusk_tweets.csv"

start_token = "<|startoftext|>"
end_token = "<|endoftext|>"
raw_text = ''
with open(file_name, 'r', encoding='utf8', errors='ignore') as fp:
    fp.readline()   # skip header
    reader = csv.reader(fp)
    for row in reader:
        raw_text += start_token + row[0] + end_token + "\n"
fp.close()

with open("intermediate_output.txt", 'w', encoding='utf8') as output_file:
    output_file.write(raw_text)
output_file.close()

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              dataset='intermediate_output.txt',
              model_name=model_name,
              steps=2000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=500,
              save_every=500,
              learning_rate=1e-5
              )

returned_data = gpt2.generate(sess, return_as_list=True, run_name='run1')

with open('output_tweets.txt','w', encoding='utf8') as output_tweets:
    for i, tweet in enumerate(returned_data):
        print('tweet number',i,':',tweet)
        output_tweets.write(str(tweet) + os.linesep)

