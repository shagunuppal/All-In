from deuces import Evaluator, Card

def tell_current_state(board_cards, hand_cards, which_cards):
    card = Card.new('Qh')
    board = []
    hand = []
    for i in range(len(board_cards)):
    	board.append(board_cards[i])
    for i in range(len(hand_cards)):
    	hand.append(hand_cards[i])
    if (which_cards == 'hand_and_board'):
    	c = Card.print_pretty_cards(board + hand)
    elif (which_cards == 'only_hand'):
    	c = Card.print_pretty_cards(hand)
    elif (which_cards == 'only_board'):
    	c = Card.print_pretty_cards(board)
    print (c)

def score(board, hand):
	evaluator = Evaluator()
	score = evaluator.evaluate(board, hand)
	print ('Score for your cards : ' + str(score) + ' / ' + str(7462))
	print ('Percentage score : ' + str(1 - (score/7462)) + ' %')
	

tell_current_state(['9s','10h','3h','7d'], ['As','Kh'],'only_board')