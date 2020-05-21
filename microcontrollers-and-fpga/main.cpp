#include <iostream>
using namespace std;

void conv()
{
	int arr[5][5] = { { 17,24,1,8,15 },{ 23,5,7,14,16 },{ 4,6,13,20,22 },{ 10,12,19,21,3 },{ 11,18,25,2,9 } };
	int filter[3][3] = { { 1,3,1 },{ 0,5,0 },{ 2,1,2 } };//only used in display
	int filterH = 3;
	int filterW = 3;
	int arrH = 5;
	int arrW = 5;
	int finish;
	int result;
	int res[5][5];
	int mcheck[9];
	for (int i = 0; i < arrH; i++)
	{
		for (int j = 0; j < arrW; j++)
		{
			cout<<arr[i][j]<<'\t';
		}
		cout << endl;
	}
	for (int i = 0; i<arrH; i++)
	{
		for (int j = 0; j< arrW; j++)
		{
			res[i][j] = 0;
				for (int m = 0; m<filterH; m++)
				{
					for (int n = 0; n<filterW; n++)
					{
						if ((((i + (filterH - 1) / 2) - m) >= 0) && (((i + (filterH - 1) / 2) - m)<arrH) && (((j + (filterW - 1) / 2) - n) >= 0) && (((j + (filterW - 1) / 2) - n)<arrW))
						{
							mcheck[m*3+n] = arr[(i + (filterH - 1) / 2) - m][(j + (filterW - 1) / 2) - n];
						}
						else
						{
							mcheck[m*3+n] = 0;
						}
					}
				}
			for (int q = 0; q<9; q++)
			{
				printf("%d\n", mcheck[q]);
			}
			for (int x = 0; x < 3; x++)
			{
				for (int y = 0; y < 3; y++)
				{
					res[i][j] += mcheck[x * 3 + y] * filter[x][y];
				}
			}
		}
	}
	
	
	for (int i = 0; i<arrH; i++)
	{
		for (int j = 0; j< arrW ; j++)
		{
			printf("%d\t", res[i][j]);
		}
		printf("\n");
	}cin >> finish;
}

int main()
{
	conv();
	return 0;
}