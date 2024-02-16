from skimage import color
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage import measure
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
from PIL import Image
import io

class ECG:
    def DividingLeads(self, image):
        image_data = image.read()

        pil_image = Image.open(io.BytesIO(image_data))

        Lead_1 = np.array(pil_image.crop((150, 300, 643, 600)))
        Lead_2 = np.array(pil_image.crop((646, 300, 1135, 600)))
        Lead_3 = np.array(pil_image.crop((1140, 300, 1625, 600)))
        Lead_4 = np.array(pil_image.crop((1630, 300, 2125, 600)))
        Lead_5 = np.array(pil_image.crop((150, 600, 643, 900)))
        Lead_6 = np.array(pil_image.crop((646, 600, 1135, 900)))
        Lead_7 = np.array(pil_image.crop((1140, 600, 1625, 900)))
        Lead_8 = np.array(pil_image.crop((1630, 600, 2125, 900)))
        Lead_9 = np.array(pil_image.crop((150, 900, 643, 1200)))
        Lead_10 = np.array(pil_image.crop((646, 900, 1135, 1200)))
        Lead_11 = np.array(pil_image.crop((1140, 900, 1625, 1200)))
        Lead_12 = np.array(pil_image.crop((1630, 900, 2125, 1200)))
        Lead_13 = np.array(pil_image.crop((150, 1250, 2125, 1480)))

        # All Leads in a list
        Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12,
                 Lead_13]
        return Leads

    def SignalExtraction_Scaling(self, Leads):
        for x, y in enumerate(Leads[:len(Leads) - 1]):
            # converting to gray scale
            grayscale = color.rgb2gray(y)
            # smoothing image
            blurred_image = gaussian(grayscale, sigma=0.7)
            # thresholding to distinguish foreground and background
            # using otsu thresholding for getting threshold value
            global_thresh = threshold_otsu(blurred_image)

            # creating binary image based on threshold
            binary_global = blurred_image < global_thresh
            # resize image
            binary_global = resize(binary_global, (300, 450))
            # finding contours
            contours = measure.find_contours(binary_global, 0.8)
            contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
            for contour in contours:
                if contour.shape in contours_shape:
                    test = resize(contour, (255, 2))

            lead_no = x
            scaler = MinMaxScaler()
            fit_transform_data = scaler.fit_transform(test)
            Normalized_Scaled = pd.DataFrame(fit_transform_data[:, 0], columns=['X'])
            Normalized_Scaled = Normalized_Scaled.T
            # scaled_data to CSV
            if os.path.isfile('scaled_data_1D_{lead_no}.csv'.format(lead_no=lead_no + 1)):
                Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no + 1), mode='a', index=False)
            else:
                Normalized_Scaled.to_csv('Scaled_1DLead_{lead_no}.csv'.format(lead_no=lead_no + 1), index=False)

    def CombineConvert1Dsignal(self):
        test_final = pd.read_csv('Scaled_1DLead_1.csv')
        location = os.getcwd()
        print(location)
        # loop over all the 11 remaining leads and combine as one dataset using pandas concat
        for files in natsorted(os.listdir(location)):
            if files.endswith(".csv"):
                if files != 'Scaled_1DLead_1.csv':
                    df = pd.read_csv('{}'.format(files))
                    test_final = pd.concat([test_final, df], axis=1, ignore_index=True)

        return test_final

    def DimensionalReduciton(self, test_final):
        pca_loaded_model = joblib.load('PCA_ECG.pkl')
        result = pca_loaded_model.transform(test_final)
        final_df = pd.DataFrame(result)
        return final_df

    def ModelLoad_predict(self, final_df):
        loaded_model = joblib.load('Heart_Disease_Prediction_using_ECG.pkl')
        result = loaded_model.predict(final_df)
        if result[0] == 1:
            return "You ECG corresponds to Myocardial Infarction"
        elif result[0] == 0:
            return "You ECG corresponds to Abnormal Heartbeat"
        elif result[0] == 2:
            return "Your ECG is Normal"
        else:
            return "You ECG corresponds to History of Myocardial Infarction"
