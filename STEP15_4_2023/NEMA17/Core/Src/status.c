#include "status.h"
#include "i2c-lcd.h"
#define PORTA GPIOA
#define DIR_PIN GPIO_PIN_1
#define STEP_PIN GPIO_PIN_2
#define M0 GPIO_PIN_5
#define M1 GPIO_PIN_4
#define M2 GPIO_PIN_3
#define PORTB GPIOB
#define Starts GPIO_PIN_11
#define Stops  GPIO_PIN_10
#define REV    GPIO_PIN_1
#define MOD1   GPIO_PIN_0
#define MOD2   GPIO_PIN_7
#define MOD3   GPIO_PIN_6
#define PRESSED  0 
int a = 0;
extern TIM_HandleTypeDef htim1;
int stepDelay = 500 ;// 1000us more delay means less speed

void microDelay (uint16_t delay)
{
  __HAL_TIM_SET_COUNTER(&htim1, 0);
  while (__HAL_TIM_GET_COUNTER(&htim1) < delay);
}

void mode1(int DOR , int RT)
{    
	 
		lcd_put_cur(1,0);
		lcd_send_string ("Full Step");

	  HAL_GPIO_WritePin(PORTA, M0, 0);
	  HAL_GPIO_WritePin(PORTA, M1, 0);
	  HAL_GPIO_WritePin(PORTA, M2, 0);
    HAL_GPIO_WritePin(PORTA, DIR_PIN, DOR);
    for(int x=0; x<(200*RT); x=x+1)
    {
      a++;
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_SET);
      microDelay(stepDelay);
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_RESET);
      microDelay(stepDelay);
			if (HAL_GPIO_ReadPin(PORTB,Stops) == PRESSED )
			{
				lcd_clear ();
				lcd_put_cur(1,0);
				lcd_send_string ("STOP");
				break;
			}

		}
}
void mode2(int DOR , int ST )
 {    
	  lcd_clear ();
		lcd_put_cur(1,0);
		lcd_send_string ("Half Step");
	  HAL_GPIO_WritePin(PORTA, M0, 1);
	  HAL_GPIO_WritePin(PORTA, M1, 0);
	  HAL_GPIO_WritePin(PORTA, M2, 0);
		int x;
    HAL_GPIO_WritePin(PORTA, DIR_PIN, DOR);
    for(x=0; x<400*ST; x=x+1)
    {
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_SET);
      microDelay(stepDelay/2);
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_RESET);
      microDelay(stepDelay/2);
			if (HAL_GPIO_ReadPin(PORTB,Stops) == PRESSED )
			{
				lcd_clear ();
				lcd_put_cur(1,0);
				lcd_send_string ("STOP");
				break;
			
      }
    }
}
 
void mode3(int DOR , int ST)
{
		lcd_clear ();
		lcd_put_cur(1,0);
		lcd_send_string ("1/32 Step");
	  HAL_GPIO_WritePin(PORTA, M0, 1);
	  HAL_GPIO_WritePin(PORTA, M1, 0);
	  HAL_GPIO_WritePin(PORTA, M2, 1);
    HAL_GPIO_WritePin(PORTA, DIR_PIN, DOR);
	
    for(int x=0; x<800*ST ; x=x+1)
    {
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_SET);
      microDelay(stepDelay/32);
      HAL_GPIO_WritePin(PORTA, STEP_PIN, GPIO_PIN_RESET);
      microDelay(stepDelay/32);
			
			if (HAL_GPIO_ReadPin(PORTB,Stops) == PRESSED )
				{
					lcd_clear ();
					lcd_put_cur(1,0);
					lcd_send_string ("STOP");
					break;			
				}
	 }
}
