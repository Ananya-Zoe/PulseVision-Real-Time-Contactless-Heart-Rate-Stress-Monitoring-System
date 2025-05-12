# mouth_tension.py

from scipy.spatial import distance as dist

class MouthTensionDetector:
    def __init__(self, upper_lip_index=62, lower_lip_index=66, max_distance=20.0):
        """
        Parameters:
        - upper_lip_index: dlib landmark index for upper inner lip
        - lower_lip_index: dlib landmark index for lower inner lip
        - max_distance: maximum distance considered for normalization
        """
        self.upper_lip_index = upper_lip_index
        self.lower_lip_index = lower_lip_index
        self.max_distance = max_distance

    def get_tension_score(self, facial_landmarks):
        """
        Calculates the mouth tension score based on lip distance.

        Parameters:
        - facial_landmarks: np.array of shape (68, 2) with dlib facial landmarks

        Returns:
        - tension_score: float in range [0.0, 1.0], where 1.0 = most tense
        """
        top_lip = facial_landmarks[self.upper_lip_index]
        bottom_lip = facial_landmarks[self.lower_lip_index]
        lip_distance = dist.euclidean(top_lip, bottom_lip)

        # Normalize (invert: closer lips = more tension)
        tension_score = 1.0 - min(lip_distance / self.max_distance, 1.0)
        return tension_score

# Debug/test
if __name__ == "__main__":
    print("MouthTensionDetector is ready to use.")
