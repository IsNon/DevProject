#include "stdint.h"
#include "stdlib.h"
#include "string.h"
#include "stm32f1xx_hal.h"
#include "status.h"
#include "i2c-lcd.h"

UART_HandleTypeDef *_huart;

static int length;
uint8_t databuff[30];
int flag;
int dir = 0;
int rota;
int Md ;

//static char** StringTok2(uint8_t* data)
//{
//	char (*ptr)[20] = (char(*)[20])malloc(10*sizeof(*ptr));
//	
//	int n = 0;
//	char *token = strtok((char*)data, " ");
//	
//	while(token != NULL)
//	{
//		strcpy(ptr[n++], token);
//		token = strtok(NULL, " "); 
//	}	
//	
//	return (char**)ptr;
//}

//static void uart_proce(int flags)
//{
//	if(flags)
//	{		
//		char** ptr = StringTok2(databuff);
//		
//		dir = atoi(ptr[0]);
//		rota = atoi(ptr[1]);
//		
//		mode3(dir, rota);

//		length = 0;
//		flag = 0;
//		memset(databuff, '\0', sizeof(databuff));
//		
//		for(uint8_t i=0; i<10; i++)
//			free(ptr[i]);
//		
//		free(ptr);
//	}
//	
//}

static char** StringTok(uint8_t* data)
{
	char** ptr = malloc(10*sizeof(char*));
	for(uint8_t i=0; i<10; i++)
	{
		ptr[i] = (char*)malloc(20*sizeof(char));
	}
	
	int n = 0;
	char *token = strtok((char*)data, " ");
	
	while(token != NULL)
	{
		strcpy(ptr[n++], token);
		token = strtok(NULL, " "); 
	}	
	
	return ptr;
}


static void uart_proce(int flags)
{
	if(flags)
	{		
		char** ptr = 	StringTok(databuff);
		
		dir = atoi(ptr[0]);
		rota = atoi(ptr[1]);
		
		mode3(dir, rota);

		length = 0;
		flag = 0;
		memset(databuff, '\0', sizeof(databuff));
		
		for(uint8_t i=0; i<10; i++)
			free(ptr[i]);
		
		free(ptr);
	}
	
}

//static void uart_proce(int flags)
//{
//	if(flags)
//	{		
//		char a[10][20];
//		int n = 0;
//		char *token = strtok((char*)databuff, " ");
//		
//		while(token != NULL)
//		{
//			strcpy(a[n++], token);
//			token = strtok(NULL, " "); 
//		}		
//		
//		dir = atoi(a[0]);
//		rota = atoi(a[1]);
//		
//		mode3(dir, rota);

//		length = 0;
//		flag = 0;
//		memset(databuff, '\0', sizeof(databuff));
//	}
//	
//}

void uart_handle(uint8_t datarx)
{
	if(datarx == '\n')
	{
		flag = 1;
	}
	
	else 
	{
		databuff[length++] = datarx;
	}
	
	uart_proce(flag);
}


void uart_init(UART_HandleTypeDef *huart1)
{
	_huart = huart1;
}

