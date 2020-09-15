// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
/*Bouncing ball.Write a small game.The window itself is the field, it is bordered by three walls(left, right, top).
The fourth side is open, there is a wallet at that side.There is also a ball in the field, its speed is constant.
It bounces back from both the walls and the racket.The racket can be controlled by mouse and/or keyboard.
If the ball falls down, the game exits.
*/

#include "SimpleGame.h"

using namespace cv;
using namespace std;

Mat image;
int iksz, ipszilon;
int ballX, ballY;
int flag;

template <typename Duration, typename Function>
void timer(Duration const & d, Function const & f)
{
	std::thread([d, f]() {
		std::this_thread::sleep_for(d);
		f();
	}).detach();
}

void borders() {
	/*left wall*/
	rectangle(image, Point(0, 0), Point(BORDERWIDTH, HEIGHT), Scalar(0, 0, 0), FILLED);
	/*right wall*/
	rectangle(image, Point(WIDTH - BORDERWIDTH, 0), Point(WIDTH, HEIGHT), Scalar(0, 0, 0), FILLED);
	/*top wall*/
	rectangle(image, Point(0, 0), Point(WIDTH, BORDERHEIGHT), Scalar(0, 0, 0), FILLED);
}

void moveBall() {
	int oldFlag = 0;
	/*if the ball hits the bottom of the screen, tha game exits*/
	if (ballY >= HEIGHT) {
		cout << "Game over!! Exiting..." << endl;
		exit(0);
	}
	/*if the ball hits the left vertical border*/
	if (ballX <= BORDERWIDTH){
		/*bounce the ball to playfield*/
		ballX = ballX + 1;
		oldFlag = flag;
		if (oldFlag == 3) {
			flag = 2;
		}
		else{
			flag = 0;
		}	
	}
	/*if the ball hits the right vertical border*/
	if (ballX >= WIDTH - BORDERWIDTH) {
		ballX = ballX - 1;
		oldFlag = flag;
		if (oldFlag == 0) {
			flag = 1;
		}
		else {
			flag = 3;
		}		
	}
	/*if the ball hits the top horizontal border*/
	if (ballY <= BORDERHEIGHT) {
		/*bounce the ball to playfield*/
		ballY = ballY + 1;
		oldFlag = flag;
		if (oldFlag == 3) {
			flag = 1;
		}
		else {
			flag = 0;
		}
	}
	/*if the ball hits the wallet*/
	if (ballY == (ipszilon - WALLETHEIGHT) && (ballX >= (iksz - WALLETWIDTH / 2) && ballX <= (iksz + WALLETWIDTH / 2))) {		
		ballY = ballY - 1;
		oldFlag = flag;
		if (oldFlag == 1) {
			flag = 3;
		}
		else {
			flag = 2;
		}
		
	}
	if (ballY == ipszilon && (ballX >= (iksz - WALLETWIDTH / 2) && ballX <= (iksz + WALLETWIDTH / 2))) {	
		ballY = ballY + 1;
		oldFlag = flag;
		if (oldFlag == 3) {
			flag = 1;
		}
		else {
			flag = 0;
		}
	}
	rectangle(image, Point(ballX - BALLWIDTH/2, ballY - BALLHEIGHT/2), Point(ballX + BALLWIDTH/2, ballY + BALLHEIGHT/2),
			  Scalar(0, 0, 0), FILLED);
	imshow("Display window", image);

	switch (flag) {
		case 0 :
			ballX = ballX + 1;
			ballY = ballY + 1;
			break;
		case 1 :
			ballX = ballX - 1;
			ballY = ballY + 1;
			break;
		case 2 :
			ballX = ballX + 1;
			ballY = ballY - 1;
			break;	
		case 3 :
			ballX = ballX - 1;
			ballY = ballY - 1;
			break;
		default :
			ballX = ballX + 1;
			ballY = ballY + 1;
	}
}

void redraw() {
	borders();
	rectangle(image, Point(iksz- WALLETWIDTH/2, ipszilon - WALLETHEIGHT/2), Point(iksz + WALLETWIDTH/2, ipszilon + WALLETHEIGHT/2),
			  Scalar(0, 0, 0), FILLED);
	imshow("Display window", image);                   // Show our image inside it.
}

void KeyboardCallback() {
	int key;
	key = waitKey(100);
	//iksz++;
	switch (key) {
	case 'o':
		iksz = iksz - 1;
		break;
	case 'p':
		iksz = iksz + 1;
		break;
	case 'q':
		ipszilon = ipszilon - 1;
		break;
	case 'a':
		ipszilon = ipszilon + 1;
		break;
	}
}

void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		iksz = x;
		ipszilon = y;
		redraw();
	}
}

int main(int argc, char** argv)
{
	image = Mat::ones(HEIGHT, WIDTH, CV_8UC3);
	flag = 0;
	iksz = WIDTH / 2;
	ipszilon = HEIGHT;
	std::random_device rd;														// obtain a random number from hardware
	std::mt19937 eng(rd());														// seed the generator
	std::uniform_int_distribution<> distrX(0, WIDTH-BALLWIDTH);					// define the range for width
	std::uniform_int_distribution<> distrY(0, HEIGHT-BALLHEIGHT);				// define the range for height
	ballX = distrX(eng);
	ballY = distrY(eng);

	rectangle(image, Point(0, 0), Point(WIDTH, HEIGHT), Scalar(255, 255, 255), FILLED);
	borders();
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display
	imshow("Display window", image);
	
	while(1) {
		rectangle(image, Point(0, 0), Point(WIDTH, HEIGHT), Scalar(255, 255, 255), FILLED);
		borders();
		/*bottom wallet*/
		rectangle(image, Point(iksz - WALLETWIDTH/2, ipszilon - WALLETHEIGHT), Point(iksz + WALLETWIDTH/2, ipszilon),
			Scalar(0, 0, 0), FILLED);
		/*make a ball that keep moving continuously at a constant speed*/
		timer(std::chrono::milliseconds(1), &moveBall);
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		waitKey(10);
		/*mouse callback for bottom wallet*/
		setMouseCallback("Display window", MouseCallBackFunc, NULL);		
		/*keyboard callback for bottom wallet*/
		KeyboardCallback();
		imshow("Display window", image);
		waitKey(10);
	}
	return 0;
}
//#endif
