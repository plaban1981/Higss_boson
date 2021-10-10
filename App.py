import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam




model = tf.keras.models.load_model('Higgs_scaled.h5')

def prediction(model,input):
    prediction = model.predict(input)
    print('prediction successful')
    return 's' if prediction[0][0] >= 0.5 else 'b'

def proba(model,input):
    proba = model.predict(input)
    print('probability successful')
    return proba


col = [['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt',
       'Weight']]

def main():
    st.header('Higgs Boson Event Detection')
    img = Image.open("higgs.JPG")
    st.image(img)

    st.write('This is a simple demo of the Streamlit framework')
    st.write('It demonstrates how to load a model, make predictions, and display the results')
    st.write('The model was trained on the Higgs Boson dataset')

    st.subheader('Input the Data')
    st.write('Please input the data below')

    i = st.number_input('DER_mass_MMC',)
    j = st.number_input('DER_mass_transverse_met_lep',)
    k = st.number_input('DER_mass_vis',)
    l = st.number_input('DER_pt_h',)
    m = st.number_input('DER_deltaeta_jet_jet',)
    n = st.number_input('DER_mass_jet_jet',)
    o = st.number_input('DER_prodeta_jet_jet',)
    a = st.number_input('DER_deltar_tau_lep',)
    b = st.number_input('DER_pt_tot',)
    c = st.number_input('DER_sum_pt',)
    d = st.number_input('DER_pt_ratio_lep_tau',)
    e = st.number_input('DER_met_phi_centrality',)
    f = st.number_input('DER_lep_eta_centrality',)
    g = st.number_input('PRI_tau_pt',)
    p = st.number_input('PRI_tau_eta',)
    r = st.number_input('PRI_tau_phi',)
    s = st.number_input('PRI_lep_pt',)
    t = st.number_input('PRI_lep_eta',)
    u = st.number_input('PRI_lep_phi',)
    v = st.number_input('PRI_met',)
    w = st.number_input('PRI_met_phi',)
    x = st.number_input('PRI_met_sumet',)
    y = st.number_input('PRI_jet_num',)
    z = st.number_input('PRI_jet_leading_pt',)
    z1 = st.number_input('PRI_jet_leading_eta',)
    z2 = st.number_input('PRI_jet_leading_phi',)
    z3 = st.number_input('PRI_jet_subleading_pt',)
    z4 = st.number_input('PRI_jet_subleading_eta',)
    z5 = st.number_input('PRI_jet_subleading_phi',)
    z6 = st.number_input('PRI_jet_all_pt',)
    z7 = st.number_input('Weight',)




    input = np.array([[i,j,k,l,m,n,o,a,b,c,d,e,f,g,p,r,s,t,u,v,w,x,y,z,z1,z2,z3,z4,z5,z6,z7]])
    print(type(i))
    print(input)
    
    
    if st.button('Detect Event'):
        pred = prediction(model,input)        
        st.success('The event is predicted is ' + pred)

    if st.button('Show Probability'):
        prob = proba(model,input)
        st.success('The probability of the event is {}'.format(prob[0][0]))

if __name__ == '__main__':

    main()

