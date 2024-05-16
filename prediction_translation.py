from keras.models import load_model
from util import *
filename="char2encoding.pkl"
sentence="We are doing language translation"
input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index= getChar2encoding(filename)
encoder_input_data=encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens)
encoder_model= load_model('encoder_modelPredTranslation.h5')
decoder_model= load_model('decoder_modelPredTranslation.h5')

input_seq = encoder_input_data

decoded_sentence=decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index)
print('-')
print('Input sentence:', sentence)
print('Decoded sentence:', decoded_sentence)


# BLEU SCORE
from nltk.translate.bleu_score import sentence_bleu

def get_bleu_score(filename):
    with open(str(filename), 'r') as f:
        lines = f.readlines()
    total_score = 0
    for line in lines:
        elements = line.strip().split('\t')
        y_true = elements[0].split()
        y_pred = elements[1].split()
        total_score += sentence_bleu([y_true], y_pred)
    return total_score / len(lines)

# ROUGE SCORE
from rouge_score import rouge_scorer

def get_rouge_score(filename):
    #Rouge1 s'enfoca més en  la similitud de les paraules individuals
    #RoueL s'enfoca més en les seqûències més llarges de paraules 
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    with open(str(filename), 'r') as f:
        lines = f.readlines()
    total_rouge1 = 0
    total_rougeL = 0
    for line in lines:
        elements = line.strip().split('\t')
        y_true = elements[0]
        y_pred = elements[1]
        scores = scorer.score(y_true, y_pred)
        total_rouge1 += scores['rouge1'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
    avg_rouge1 = total_rouge1 / len(lines)
    avg_rougeL = total_rougeL / len(lines)
    return avg_rouge1, avg_rougeL
