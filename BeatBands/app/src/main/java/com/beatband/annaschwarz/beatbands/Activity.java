
package com.beatband.annaschwarz.beatbands;

import android.content.Intent;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.telephony.SmsManager;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import android.net.Uri;
import android.widget.TextView;
import android.app.AlertDialog;
import android.content.DialogInterface;

public class Activity extends android.app.Activity {

    TextView mTextField;
    boolean click = false;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.emergency);

        Button button1 = (Button) findViewById(R.id.button1);
        Button button2 = (Button) findViewById(R.id.button2);
        Button button3 = (Button) findViewById(R.id.button3);
        mTextField = (TextView) findViewById((R.id.textView));



        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                click = true;
                AlertDialog.Builder builder = new AlertDialog.Builder(Activity.this);
                builder.setMessage("Are you sure you are ok?")
                        //.setCancelable(false)
                        .setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                finish();

                            }
                        })
                        .setNegativeButton("No", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id)
                            {
                                dialog.cancel();
                            }
                        });
                AlertDialog alert = builder.create();
                alert.show();
            }

            });


        //txts a friend

        button2.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view){
            click = true;
                Text();
            Home();
            }
        });

        //calls a friend

        button3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                click = true;
                Intent phoneIntent = new Intent(Intent.ACTION_CALL);
                phoneIntent.setData(Uri.parse("tel:443-676-8571"));
                startActivity(phoneIntent);
                //Home();
            }
        });

        new CountDownTimer(10000, 1000) {

            long n;

            public void onTick(long millisUntilFinished) {
                n = millisUntilFinished / 1000;
        if(click==true){
            n=0;
                }
                mTextField.setText("seconds remaining: " + millisUntilFinished / 1000);

        if(n == 5){
                    Text();
                }

                if(n == 1){
                    Intent phoneIntent = new Intent(Intent.ACTION_CALL);
                    phoneIntent.setData(Uri.parse("tel:443-676-8571"));
                    startActivity(phoneIntent);

                }
    }

    public void onFinish() {
                Home();
            }


        }.start();





    }
    private void Home(){
        Intent back = new Intent(Activity.this, HomeActivity.class);
        startActivity(back);
    }

    private void Text(){
        SmsManager smsManager = SmsManager.getDefault();
        smsManager.sendTextMessage("4437999993", null, "Your Friend is in danger!", null, null);
        Toast.makeText(getApplicationContext(),"A message has been sent to your friend.",Toast.LENGTH_LONG).show();
    }



}