#include <iostream>
using namespace std;

// 函数申明
int max(int num1,int num2);

int main(){

    // 局部变量申明
    int a =100;
    int b = 200;
    int ret;

    ret = max(a,b);

    cout << "Max value is :" << ret << endl;
    return 0;
}

int max(int num1,int num2){

    int result;

    if(num1>num2){
        result = num1;
    }
    else{
        result = num2;
    }

    return result;

}