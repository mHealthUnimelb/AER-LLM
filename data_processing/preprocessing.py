import os
from pathlib import Path


def get_dialog_order(tran_path):
    diaglog_ids = []
    for session in range(1,6):
        session = "Session" + str(session)
        session_path = os.path.join(tran_path, session)
        trans_path = os.path.join(session_path, 'dialog/transcriptions')
        trans_lists = sorted(os.listdir(trans_path))
        for i in range(len(trans_lists)):
            full_trans_path = os.path.join(trans_path, trans_lists[i])
            texts = open(full_trans_path, 'r').readlines()
            for text in texts:
                sentence_id = text.split()[0]
                diaglog_ids.append(sentence_id)
    return diaglog_ids

def get_label_prob(data):

    emo_labels = ['neu', 'hap', 'ang', 'sad']
    emotion_code_dict = {"Neutral state":"neu", "Happiness":"hap", "Anger":"ang", "Sadness":"sad", "Frustration":"others", "Contempt":"others", "Excitement":"others", "Surprise":"others", "Disgust":"others", "Fear":"others", "Other": "others"}
    num_out_labels = 0
    for item in data:
        amb_labels = []
        if item['need_prediction'] == 'yes':
            for emo in item['emotion']:
                amb_labels.append(emotion_code_dict[emo])

            filtered_labels = [label for label in amb_labels if label in emo_labels]
            for label in amb_labels:
                if label not in emo_labels:
                    num_out_labels += 1

            item['amb_emotion'] = filtered_labels

            emotion_counts = Counter(filtered_labels)
            total_count = sum(emotion_counts.values())
            
            probs = {emo: round(emotion_counts[emo]/total_count,2) for emo in emo_labels}
            item['emotion_probs'] = [probs[emo] for emo in emo_labels]

    return data, num_out_labels

# rearrange the data into sessions
def rearrange_data(data):
    session_dict = {}
    for item in data:
        session_impro_id = item['id'][:6]
        if session_impro_id not in session_dict:
            session_dict[session_impro_id] = []
        session_dict[session_impro_id].append(item)
    return session_dict

def order_sentences(diaglog_id_orders, new_data):
    order_sen = []
    for sentence_id in diaglog_id_orders:
        for item in new_data:
            if item['id'] == sentence_id:
                order_sen.append(item)
                break
    return order_sen

def match_features(index, order_audio_data):
    matched_features = {}
    if 'egemaps' not in order_audio_data[index]:
        print("Error occurred: No egemaps in the audio data, the index is ", index)
    else:
        egemaps = order_audio_data[index]['egemaps']
        column_names = egemaps.columns
        values = egemaps.values[0]
        for col_id in range(len(column_names)):
            matched_features[column_names[col_id]] = float(values[col_id])
        return matched_features

def get_label_dict(data):
    for i in range(len(data)):
        item = data[i]
        if "emotion" in item.keys():
            emotion_counts = Counter(item['emotion'])
            total_count = sum(emotion_counts.values())
            probs = {emo: round(emotion_counts[emo]/total_count,2) for emo in emotion_counts}
            item['emotion_dict'] = probs
        else:
            item['emotion_dict'] = None
    return data