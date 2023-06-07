import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

def strike_zones(player_data):
    fig, ax = plt.subplots()
    player_data['type'] = player_data['type'].map({'S': 1, 'B': 0})
    player_data = player_data.dropna(subset=['type', 'plate_x', 'plate_z'])
    plt.scatter(player_data.plate_x, player_data.plate_z, c=player_data.type, cmap=plt.cm.coolwarm, alpha=0.25)
    training_set, validation_set = train_test_split(player_data, random_state=1)
    classifier = SVC(kernel='rbf', gamma=3, C=1)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))
    draw_boundary(ax, classifier)
    ax.set_ylim(-2, 6)  # Set y-axis limits
    ax.set_xlim(-3, 3)  # Set x-axis limits
    plt.show()

strike_zones(aaron_judge)  # Plot Aaron Judge's strike zone
strike_zones(jose_altuve)  # Plot Jose Altuve's strike zone
