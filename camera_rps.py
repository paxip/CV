import random
import cv2
from keras.models import load_model
import numpy as np
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)



class RPS:

    def __init__(self, game_list):
        self.game_list = game_list

       
    def get_winner(self, computer_choice, user_choice):

        if user_choice == computer_choice:
            winner = 'Draw'
            
        elif user_choice == 'Rock' and computer_choice == 'Scissors':
            winner = 'User'

        elif user_choice == 'Rock' and computer_choice == 'Paper':
            winner = 'Computer'
    
        elif user_choice == 'Paper' and computer_choice == 'Rock':
            winner = 'User'
            
        elif user_choice == 'Paper' and computer_choice == 'Scissors':
            winner = 'User'
    
        elif user_choice == 'Scissors' and computer_choice == 'Rock':
            winner = 'Computer'

        elif user_choice == 'Scissors' and computer_choice == 'Paper':
            winner = 'User'

        return winner
            
   
    def get_prediction(self):
        while True: 
            ret, frame = cap.read()
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            data[0] = normalized_image
            prediction = model.predict(data)
            cv2.imshow('frame', frame)
            # Press q to close the window
            print(prediction)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        
        return prediction
        


    def get_computer_choice(self):
        computer_choice = random.choice(game_list)
        return computer_choice


def play(game_list):
    game = RPS(game_list)
    computer_choice = game.get_computer_choice() 
    user_choice = game.get_user_choice()
    winner = game.get_winner(computer_choice, user_choice)
    if winner == 'Draw':
        print(f'Both players selected {user_choice}, it is a tie.')
    elif winner == 'User':
        print('User wins')
    else:
        winner == 'Computer'
        print('Computer wins')

if __name__ == '__main__':
    game_list = ['Rock', 'Paper', 'Scissors']
    play(game_list)
