import random
import torch
import math
from torch.nn.utils.rnn import pad_sequence

def countTrainableParameters(model) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params



def convert_ans_to_token(answer, label2id, max_seq_length = 512 ):

  ## Simple Trick to pad a sequence to deired length
  dummy_array = torch.zeros(max_seq_length)
  actual_ans_array = []

  answer = answer.split(" ")
  for token in answer:
    actual_ans_array.append(label2id[token]['id'])
  
  actual_ans_array = torch.tensor(actual_ans_array, dtype = torch.int32)
  actual_ans_array = pad_sequence([actual_ans_array,dummy_array], batch_first  = True)[0]

  return actual_ans_array


def convert_ques_to_token(question, tokenizer, pad_token_id = 0, max_seq_len = 512):

  question_array = []
  question = question.split(" ")
  
  for token in question:
    question_array.extend(tokenizer(token, add_special_tokens = False).input_ids)
  
  if len(question_array)< max_seq_len:
        question_array.extend([pad_token_id]* (max_seq_len-len(question_array)))

  question_array = torch.tensor(question_array, dtype = torch.int32)
  return question_array[:max_seq_len]



def convert_token_to_ques(ques, tokenizer):
  decoded_ques = tokenizer.decode(ques, skip_special_tokens=True)
  return decoded_ques


def convert_token_to_answer(ans, id2label):
  non_zero_argument = torch.nonzero(ans,as_tuple = False).view(-1)

  actual_answer = ans[non_zero_argument].cpu().numpy()
  decoded_answer = []
  
  for token in actual_answer:
    decoded_answer.append(id2label[token])
  
  decoded_answer = " ".join(decoded_answer)
  return decoded_answer

