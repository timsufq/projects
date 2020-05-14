#include <iostream>
using namespace std;

int main()
{
	int exit = 0;
	int *a = &exit;
	cout << "a: " << a << endl;
	cout << "*a: " << *a << endl;
	cout << "&a: " << &a << endl;
	cin >> exit;
	return 0;
}

// After "a" is declared as a pointer variable by the "int *a" statement, "a" is treated as an independent variable, 
// which has its own address and content, and the address stored in "a"'s content is independent from "a"'s own address.
// Meaning of different terms:
// a: The content of "a", and this content is an address which is independent from "a"'s own address.
// *a: The content within the address which is "a"'s content.
// &a: The address of "a" variable, which is independent from "a"'s content.
