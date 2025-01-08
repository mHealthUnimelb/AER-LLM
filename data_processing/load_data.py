import os
from pathlib import Path
from data_processing import *
import pickle

def IEMO_data()
    # Get the dialog order for all sessions
    tran_path = './data/IEMOCAP_full_release/'
    diaglog_id_orders = get_dialog_order(tran_path)
    # Load text data
    file_path = './data/iemocap_ambiguous.json'
    data = json.load(open(file_path))
    # load audio data
    with open('./data/iemocap_egemaps.pkl', 'rb') as f:
    audio_features = pickle.load(f, encoding='latin1')

    # data processing
    # Transfer three ground true labels to distributions
    new_data = get_label_prob(data)
    # Rearrange data into sessions:
    session_data = rearrange_data(new_data)
    # Reorder data according to dialog ids
    order_data = order_sentences(diaglog_id_orders, new_data)
    order_audio_data = order_sentences(diaglog_id_orders, audio_features)

    return order_data, order_audio_data


def msp_data():
    msp = json.load(open('./data/msp_ambigous.json'))
    with open('./data/msp_egemaps.pkl', 'rb') as f:
        audio_df = pickle.load(f)

    #selected_audio_df = {ID: audio_df[ID] for ID in index}
    selected_audio_df = [audio_df[ID] for ID in index]

    msp, num_out_labels = get_label_prob(msp)

    msp = get_label_dict(msp)

    started_sessions = []
    organ_msp = {}
    for i in range(len(msp)):
        podcast_ID = "_".join(msp[i]['id'].split("_")[0:2])
        if podcast_ID not in started_sessions:
            started_sessions.append("_".join(msp[i]['id'].split("_")[0:2]))
            organ_msp[podcast_ID] = []
        organ_msp[podcast_ID].append(msp[i])

    audio_id_in_podcast = {}
    num = 0
    for podcast_ID in organ_msp.keys():
        audio_id_in_podcast[podcast_ID] = {}
        for i in range(len(organ_msp[podcast_ID])):
            if organ_msp[podcast_ID][i]['need_prediction'] == 'yes':
                audio_maps = selected_audio_df[num]
                num += 1
                audio_id_in_podcast[podcast_ID][i] = audio_maps

    need_podcast_id = []

    for ID, podcast in organ_msp.items():
        for i in range(len(podcast)):
            if podcast[i]["need_prediction"] == "yes" and ID not in need_podcast_id:
                need_podcast_id.append(ID)

    return need_podcast_id, audio_id_in_podcast, organ_msp

