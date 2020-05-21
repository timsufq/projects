#include <avr/io.h>
#include <avr/interrupt.h>

void t0_init(void)
{
	TCCR0A=(0<<COM0A1)|(0<<COM0A0)|(0<<COM0B1)|(0<<COM0B0)|(0<<WGM01)|(0<<WGM00);
	TCCR0B=(0<<WGM02)|(0<<CS02)|(0<<CS01)|(0<<CS00);
	TCNT0=0;
	OCR0A=0;
	OCR0B=0;
}
void t1_init(void)
{
	TCCR1A=(0<<COM1A1)|(0<<COM1A0)|(0<<COM1B1)|(0<<COM1B0)|(0<<WGM11)|(0<<WGM10);
	TCCR1B=(0<<WGM13)|(0<<WGM12)|(0<<CS12)|(0<<CS11)|(0<<CS10);
	TCNT1=0;
	OCR1A=0;//15625 for 1 second
	OCR1B=0;
}
void t2_init(void)
{
	TCCR2A=(0<<COM2A1)|(0<<COM2A0)|(0<<COM2B1)|(0<<COM2B0)|(0<<WGM21)|(0<<WGM20);
	TCCR2B=(0<<WGM22)|(0<<CS22)|(0<<CS21)|(0<<CS20);
	TCNT2=0;
	OCR2A=0;
	OCR2B=0;
}
void interrupt_init(void)
{
	//TIMSK0=0x00;//TIMSK1;TIMSK2//Timer Interrupt
	//EIMSK=(1<<INT0);//INT1//External Interrupt
	//EICRA=(1<<ISC01)|(1<<ISC00);//ISC11;ISC10, both 1 for rising edge//External Interrupt
	sei();
	//ADCSRA|=(1<<ADSC);//Start ADC convert//add this to the end of ADC interrupt ISR for start a new one
}
void usart_init(void)
{
	UBRR0L=0x67;//67 for 9600
	UCSR0B=(1<<TXEN0);//|(1<<RXEN0)|(1<<UDRIE0)|(1<<RXCIE0);
	UCSR0C=(1<<UCSZ01)|(1<<UCSZ00)|(1<<UMSEL01);//UCSZ0X=11 for 8 bit;UMSEL01=1 for Synchronous Operation
}
void adc_init(void)
{
	ADMUX=(0<<REFS1)|(1<<REFS0)|(1<<ADLAR)|(0<<MUX1)|(0<<MUX0);//ADCH is used when ADLAR=1
	ADCSRA=(1<<ADEN)|(1<<ADIE)|(1<<ADPS2)|(1<<ADPS1)|(1<<ADPS0);
}
int main(void)
{
	//DDRB=0xFF;
	//DDRC=0xFF;
	//DDRD=0xFF;
	//Insert init here, interrupt_init should be the last
	while(1){}
	return 0;
}