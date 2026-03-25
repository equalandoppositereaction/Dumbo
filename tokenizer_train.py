'''
This file was made to directly run on a virtual machine
'''

from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train"
)

import sentencepiece as spm
import os

model_prefix = "fwedu32k"
vocab_size = 40960    #512x80
model_type = "bpe"

#Add eos & bos token at the end of every row, i.e. I am stupid for listening to clankers
#this is the only place where I fully trused a clanker and this is the worst thing to exist in the  whole codebase
def text_iterator(dataset):
    for i, row in enumerate(dataset):
        combined_text = "<|beginoftext|>" + row["text"] + "<|endoftext|>" #there is absolutely no use of putting these 2 tokens here but I have no idea why I even did this
        yield combined_text

spm.SentencePieceTrainer.Train(
    sentence_iterator=text_iterator(dataset),
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=0.995, #the default 0.995 would have been better 
    byte_fallback=True,
    split_by_unicode_script=True,
    split_by_number=True,
    bos_piece='<|beginoftext|>',
    pad_piece='<|pad|>',
    unk_piece='<|unk|>',
    eos_piece='<|endoftext|>',
    user_defined_symbols=["'s", "'t", "'re", "'ve", "'m", "'ll", "'d",      #tried to replicate the regex line fropm gpt4
                            "<|user|>", "<|system|>", "<|assistant|>", "<|im_start|>", "<|im_end|>",
                            "<|thought|>", "</|thought|>","<|search_query|>", "</|search_query|>", "<|search_results|>","</|search_results|>", "\n",
                            "<|extra_0|>", "<|extra_1|>", "<|extra_2|>", "<|extra_3|>", "<|extra_4|>"
                           ],
    remove_extra_whitespaces=False,
    max_sentence_length=800000,
    train_extremely_large_corpus=True,
    input_sentence_size=2500000,       #Samples 2 million lines
    shuffle_input_sentence=True,
    num_threads=16                      #config of the VM
    )

print(f"Model and vocab saved as {model_prefix}.model and {model_prefix}.vocab")


