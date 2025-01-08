import re 
import ast 
import time

# transfer response from ```json\n{'neutral': 0.0, 'happy': 0.0, 'angry': 0.0, 'sad': 1.0}\n```
def identify_format(text):
    match = re.search(r"\{.*\}", text)
    if match:
        text = match.group(0)
    result_dict = ast.literal_eval(text)
    return result_dict
def dictToList(dict, emo_labels):
    prob_list = [dict[emo] for emo in emo_labels]
    return prob_list

log = {}
def predict_sentence_IEMO(index_sentence, number_of_contexts, data, shot_type, order_audio_data):
    error = False
    emo_labels = ['neutral', 'happy', 'angry', 'sad']
    cur_sentence = data[index_sentence]
    need_pred = cur_sentence['need_prediction']
     # if an emotion prediction is required for this sentence
    if index_sentence >= number_of_contexts:
        firstflag = False
        cur_context = data[index_sentence-number_of_contexts:index_sentence]
    else:
        firstflag = True
        cur_context = data[:index_sentence]
    cur_label = cur_sentence['emotion_probs']
    # context is all previous sentences in the conversation
    if "egemaps" in order_audio_data[index_sentence].keys():
        audio2text = match_features(index_sentence,order_audio_data)
    else:
        audio2text = ""
    try:
        time.sleep(0.1)
        response,prompt = Gemini_emotion_predictor(cur_context, cur_sentence, number_of_contexts, firstflag,audio2text, shot_type)
        response = response.text.strip()
        # input both context and the current sentence to the emotion predictor
        try:
            clear_response = identify_format(response)
            cur_pred = dictToList(clear_response, emo_labels)
            log[cur_sentence["id"]] = [prompt, response]
        except:
            # if there is an error, fill a neutral to keep the output of same dimension
            print('Gemini response is not in the right format: ', response, cur_sentence['id'])
            cur_pred = [1.0,0.0,0.0,0.0]
            error = True
            log[cur_sentence["id"]] = ["Response not in the right format", prompt, response]
    except:
        print('Gemini api has an error.: ', cur_sentence)
        cur_pred = [1.0,0.0,0.0,0.0]
        error = True
        log[cur_sentence["id"]] = ["Gemini api has an error."]

    return cur_label, cur_pred, error 

def make_predictions_IEMO(number_context, shot_type, order_audio_data, order_data):
    number_errors, number_success = 0, 0
    started_sessions = []
    all_ground_truth, all_pred = [], []

    for i, item in enumerate(order_data):
        
        if item['id'][:-4] not in started_sessions:
            started_sessions.append(item['id'][:-4])
            print("Session ", item['id'][:-4])
            
        if item["need_prediction"] == "yes":
            label, prediction, error = predict_sentence(i, number_context, order_data, shot_type, order_audio_data)
            all_ground_truth.append(label)
            all_pred.append(prediction)
            if error == True:
                number_errors += 1
            else: 
                number_success += 1

        if i == len(order_data)-1 or order_data[i+1]['id'][:-4] != order_data[i]['id'][:-4]:
            print('Number of error counts:', number_errors, "; Number of predictions:", number_success)
            number_errors, number_success = 0, 0
            print('------------------------')
        # for testing
        # if i > 50:
        #     break
    return all_pred, all_ground_truth


def predict_sentence_msp(index_sentence, number_of_context, data, podcast_I, shot_type, audio_id_in_podcast):
    error = False
    emo_labels = ['neutral', 'happy', 'angry', 'sad']
    cur_sentence = data[index_sentence]
    cur_label = cur_sentence['emotion_probs']
    if index_sentence >= number_of_context:
        firstflag = False
        cur_context = data[index_sentence-number_of_context:index_sentence]
    else:
        firstflag = True
        cur_context = data[:index_sentence]

   # context is all previous sentences in the conversation
    audio2text = match_features(index_sentence, audio_id_in_podcast[podcast_ID])
        
    try:
        time.sleep(0.05)
        response,prompt = Gemini_emotion_predictor(cur_context, cur_sentence, number_of_context, firstflag, audio2text, shot_type)
        response = response.text.strip()

        # input both context and the current sentence to the emotion predictor
        try:
            clear_response = identify_format(response)
            cur_pred = dictToList(clear_response, emo_labels)
            log[cur_sentence["id"]] = [prompt, response]
        except:
            # if there is an error, fill a neutral to keep the output of same dimension
            print('Gemini response is not in the right format: ', response, cur_sentence['id'])
            cur_pred = [1.0,0.0,0.0,0.0]
            error = True
            log[cur_sentence["id"]] = ["Response not in the right format", prompt, response]
    except:
        try: 
            time.sleep(5)
            response,prompt = Gemini_emotion_predictor(cur_sentence)
            response = response.text.strip()

            # input both context and the current sentence to the emotion predictor
            try:
                clear_response = identify_format(response)
                cur_pred = dictToList(clear_response, emo_labels)
                log[cur_sentence["id"]] = [prompt, response]
            except:
                # if there is an error, fill a neutral to keep the output of same dimension
                print('Gemini response is not in the right format: ', response, cur_sentence['id'])
                cur_pred = [1.0,0.0,0.0,0.0]
                error = True
                log[cur_sentence["id"]] = ["Response not in the right format", prompt, response]

        except:
            print('Gemini api has an error.: ', cur_sentence)
            cur_pred = [1.0,0.0,0.0,0.0]
            error = True
            log[cur_sentence["id"]] = ["Gemini api has an error."]

    return cur_label, cur_pred, error 

log = {}
def make_predictions_msp(data, number_of_context, need_podcast_id, shot_type, audio_id_in_podcast):
    number_errors, number_success = 0, 0
    started_sessions = []
    all_ground_truth, all_pred = [], []
    
    for podcast_ID in need_podcast_id:
        podcast = data[podcast_ID]
        if podcast_ID not in started_sessions:
            started_sessions.append(podcast_ID)
            print("Podcast ", podcast_ID)
        
        for i in range(len(podcast)):
            if podcast[i]["need_prediction"] == "yes":
                
                label, prediction, error = predict_sentence(i,number_of_context, podcast,podcast_ID, shot_type, audio_id_in_podcast)

                all_ground_truth.append(label)
                all_pred.append(prediction)
                if error == True:
                    number_errors += 1
                else: 
                    number_success += 1

        print('Number of error counts:', number_errors, "; Number of predictions:", number_success)
        number_errors, number_success = 0, 0
        print('------------------------')
        # for testing
        # if i > 50:
        #     break
    print("Total predictions: ", len(all_pred), "Total ground truth:", len(all_ground_truth))

    return all_pred, all_ground_truth