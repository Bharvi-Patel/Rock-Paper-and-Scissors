import cv2
import numpy as np
import random
import time
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('rps_model.keras')

# Define gesture labels
gestures = ['rock', 'paper', 'scissors']

# Game rules
def get_winner(player_move, computer_move):
    if player_move == computer_move:
        return "Draw"
    elif (player_move == 'rock' and computer_move == 'scissors') or \
         (player_move == 'paper' and computer_move == 'rock') or \
         (player_move == 'scissors' and computer_move == 'paper'):
        return "You Win!"
    else:
        return "Computer Wins!"

# Start video capture
cap = cv2.VideoCapture(0)

# Score tracking
rounds = 5
player_score = 0
computer_score = 0
round_count = 0

while round_count < rounds:
    # Show live feed to place hand
    start_time = time.time()
    while time.time() - start_time < 3:  # Allow 3 seconds for hand placement
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # Draw ROI box
        height, width, _ = frame.shape
        box_width, box_height = 200, 200
        x1, y1 = width - box_width - 20, height - box_height - 20
        x2, y2 = width - 20, height - 20
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, 'Place Hand Here', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Countdown display
        countdown = 3 - int(time.time() - start_time)
        cv2.putText(frame, f'Capturing in {countdown}...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Rock Paper Scissors Game', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Capture frame for prediction
    ret, original_frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(original_frame, 1)  # For display to user

    # Define ROI from flipped frame 
    height, width, _ = frame.shape
    box_width, box_height = 200, 200
    x1, y1 = width - box_width - 20, height - box_height - 20
    x2, y2 = width - 20, height - 20

    # Convert box coordinates from flipped frame to original frame
    x1_orig = original_frame.shape[1] - x2
    
    x2_orig = original_frame.shape[1] - x1

# Use original (unflipped) frame for prediction
    roi = original_frame[y1:y2, x1_orig:x2_orig]
    roi = cv2.flip(roi, 1) 

# Preprocess for model
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Optional but good practice
    img = cv2.resize(roi_rgb, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    player_move = gestures[np.argmax(prediction[0])]

    computer_move = random.choice(gestures)
    result = get_winner(player_move, computer_move)

    if result == "You Win!":
        player_score += 1
    elif result == "Computer Wins!":
        computer_score += 1

    round_count += 1

    # Display result
    cv2.putText(frame, f'Round {round_count} of {rounds}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f'Your Move: {player_move}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(frame, f'Computer: {computer_move}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(frame, f'Result: {result}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Score: You {player_score} - {computer_score} Computer', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow('Rock Paper Scissors Game', frame)


    # Pause for 3 seconds between rounds
    for i in range(5, 0, -1):
        countdown_frame = frame.copy()
        cv2.putText(countdown_frame, 'Next round...', (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.imshow('Rock Paper Scissors Game', countdown_frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final result
final_result = "You Win the Game!" if player_score > computer_score else \
               "Computer Wins the Game!" if computer_score > player_score else "It's a Draw!"

cv2.putText(frame, final_result, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
cv2.imshow('Rock Paper Scissors Game', frame)
cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()