#include "stm32f10x.h"
#include "PinMap.h"
#include "stdio.h"
#include "misc.h"

void DelayMs(uint32_t ms);
static __IO uint32_t msTicks;

int main(void)
{
	GPIO_WriteBit(GPIOA, GPIO_Pin_5, Bit_SET);
	GPIO_WriteBit(GPIOA, GPIO_Pin_5, Bit_RESET);
	while(true){}
	return 0;
}
void DelayMs(uint32_t ms)
{
	msTicks=ms;
	while(msTicks);
}
void SysTick_Handler()
{
	if(msTicks!=0)
	{
		msTicks--;
	}
}
