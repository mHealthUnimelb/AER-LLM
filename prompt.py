
def prompt_context(firstflag, context, cur_sentence, audio2text, shot_type):
    
    background = 'Two speakers are talking. The conversation is:\n'
    if firstflag == False:
        context = '\n'.join(f"{item['speaker']}: {item['groundtruth']}" for item in context[-number_of_contexts:]) + '\n'
    else:
        context = '\n'.join(f"{item['speaker']}: {item['groundtruth']}" for item in context) + '\n'
    new_sentence = f"Now speaker {cur_sentence['speaker']} says: '{cur_sentence['groundtruth']}'. \n"
    cur_context = context[-number_of_contexts:]

    example = "Examples: \n" + "\n".join(f"Sentence {i+1}: {cur_context[i]['speaker']}: {cur_context[i]['groundtruth']} Emotion probabilities: {cur_context[i]['emotion_dict']}" for i in range(len(cur_context))) + '\n'

    task = f"Predict the probability of the emotion of the sentence '{cur_sentence['groundtruth']}' from the options [neutral, happy, angry, sad], consider the conversation context. Output statisfies the following rules.\n"
    requirement = "Output satisfies the following rules. \n" \
                "Rule 1: Consider the examples.\n" \
                 "Rule 2: Generate a dictionary of emotion probabilities in format of {'neutral': 0.1, 'happy':0.0, 'angry':0.1, 'sad':0.8} and only consider these four emotions." \
                         "If you think there is only one emotion in the sentence, then give the probability to 1.\n" \
                "Rule 3: Ensure the sum of probability equal to 1.\n" 
    task_amb2 = "Rule 4: Do not explain, only the dictionary.\n"
    task_final = "Please check again whether your output follows the three rules."

    if audio2text != "":
        audio = f"Here are 88 audio features of the current speaker's sentence, please find useful information from the features to finish a task. The features are: {audio2text}"
    else:
        audio = " "
    if shot_type == 'zs':  
        prompt = background + context + new_sentence + audio + task + requirement + task_amb2 + task_final
    elif shot_type == 'fs':
        prompt = background + context + new_sentence + audio  + task + example + requirement + task_amb2 + task_final

    return prompt 